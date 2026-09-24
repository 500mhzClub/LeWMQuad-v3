"""Rank-regret metric-validity study V1.

Implements the registered study of
``docs/lewm_go2_rank_regret_metric_validity_v1_preregistration_2026-08-05.md``.

Two distinct one-step metrics are in play and this module keeps them separate
throughout:

``G``
    Geometric first-action regret in metres: progress lost relative to the best
    available branch.  This is what the closed-loop harness measured.
``R``
    Normalized rank regret, dimensionless: the dense-rank position of the
    selected branch over ``max(1, max_dense_rank)``.  This is what the fixed
    gates of handoff sections 11 and 13 used.

Part B1 correlates ``G`` against closed-loop progress across the bound Aug-4
policies.  Part B2 correlates ``R`` against ``G`` on the immutable matched
panel, same-state, removing the trajectory-divergence confound.

The module has no filesystem, RGB, simulator, or encoder access.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from lewm.benchmarks import go2_observability_ceiling_assay_v1 as assay


SCHEMA = "lewm_go2_rank_regret_metric_validity_v1"

BOOTSTRAP_SEED = 2_026_080_504
BOOTSTRAP_RESAMPLES = 10_000

# Registered decision thresholds (preregistration section 6).
CLOSED_LOOP_LINK_RHO = -0.7
CLOSED_LOOP_LINK_BOUND = -0.3
RANK_LINK_RHO = 0.7
RANK_LINK_BOUND = 0.3

VALID_PROXY = "VALID_PROXY"
INVALID_AT_RANK_LINK = "INVALID_PROXY_AT_RANK_LINK"
INVALID_AT_CLOSED_LOOP_LINK = "INVALID_PROXY_AT_CLOSED_LOOP_LINK"
AMBIGUOUS = "AMBIGUOUS"

# Part B1: values bound from the completed, independently recomputed Aug-4
# registered results.  These are transcribed constants, never recomputed here.
BOUND_CLOSED_LOOP_POLICIES: tuple[dict[str, Any], ...] = (
    {"policy": "bearing", "progress_m": 0.9000, "successes": 24, "geometric_regret_m": 0.02486},
    {"policy": "oracle_mpc", "progress_m": 0.8151, "successes": 14, "geometric_regret_m": 0.00000},
    {"policy": "dino_true_successor", "progress_m": 0.6494, "successes": 14, "geometric_regret_m": 0.05956},
    {"policy": "random", "progress_m": 0.4968, "successes": 0, "geometric_regret_m": 0.07346},
    {"policy": "dino_true_successor_shuffled", "progress_m": 0.3537, "successes": 0, "geometric_regret_m": 0.07530},
    {"policy": "dino_persistence", "progress_m": 0.0000, "successes": 0, "geometric_regret_m": 0.13534},
    {"policy": "hold", "progress_m": 0.0000, "successes": 0, "geometric_regret_m": 0.13534},
)

BOUND_SOURCES = (
    "docs/lewm_go2_planner_oracle_assay_v1_result_2026-08-04.md",
    "docs/lewm_go2_dino_true_successor_goal_cost_v1_result_2026-08-04.md",
)


class MetricValidityError(RuntimeError):
    """Raised when the metric-validity contract is violated."""


def canonical_bytes_v1(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


# --------------------------------------------------------------------------
# Correlation and uncertainty
# --------------------------------------------------------------------------


def _ranks(values: Sequence[float]) -> np.ndarray:
    """Average ranks, matching the registered tie handling."""

    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    ranks[order] = np.arange(1, len(array) + 1, dtype=np.float64)
    # Average ranks within tied groups.
    unique, inverse, counts = np.unique(array, return_inverse=True, return_counts=True)
    for index, count in enumerate(counts):
        if count > 1:
            mask = inverse == index
            ranks[mask] = ranks[mask].mean()
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_v1(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 3:
        raise MetricValidityError("correlation requires at least three paired points")
    return _pearson(_ranks(x), _ranks(y))


def bootstrap_correlation_v1(
    x: Sequence[float], y: Sequence[float], *, seed: int = BOOTSTRAP_SEED
) -> dict[str, object]:
    """Resample scorers with replacement; report the percentile interval.

    Resamples that degenerate to a constant in either coordinate carry no rank
    information and are skipped rather than counted as zero.
    """

    point = spearman_v1(x, y)
    array_x = np.asarray(x, dtype=np.float64)
    array_y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        picked = rng.integers(0, len(array_x), size=len(array_x))
        sample_x, sample_y = array_x[picked], array_y[picked]
        if sample_x.std() == 0.0 or sample_y.std() == 0.0:
            continue
        draws.append(spearman_v1(sample_x, sample_y))
    if len(draws) < BOOTSTRAP_RESAMPLES // 10:
        raise MetricValidityError("bootstrap degenerated for this scorer panel")
    values = np.asarray(draws, dtype=np.float64)
    lower, upper = (float(item) for item in np.quantile(values, (0.025, 0.975)))
    return {
        "spearman": float(point),
        "pearson": _pearson(x, y),
        "ci_lower": lower,
        "ci_upper": upper,
        "resamples": BOOTSTRAP_RESAMPLES,
        "usable_resamples": len(draws),
        "points": len(array_x),
    }


# --------------------------------------------------------------------------
# Part B1
# --------------------------------------------------------------------------


def part_b1_v1() -> dict[str, object]:
    """Correlate bound geometric regret against bound closed-loop progress."""

    regret = [row["geometric_regret_m"] for row in BOUND_CLOSED_LOOP_POLICIES]
    progress = [row["progress_m"] for row in BOUND_CLOSED_LOOP_POLICIES]
    correlation = bootstrap_correlation_v1(regret, progress)
    return {
        "description": (
            "does geometric first-action regret in metres predict closed-loop "
            "progress across the bound Aug-4 policies"
        ),
        "bound_sources": list(BOUND_SOURCES),
        "policies": [dict(row) for row in BOUND_CLOSED_LOOP_POLICIES],
        "correlation": correlation,
        "confound": (
            "both endpoints come from the same runs, so geometric regret here is "
            "measured on each policy's own diverged states"
        ),
    }


# --------------------------------------------------------------------------
# Part B2
# --------------------------------------------------------------------------


def progress_matrix_v1(groups: Sequence[Any]) -> np.ndarray:
    """Return the ``(states, actions)`` target-progress matrix in metres."""

    rows = []
    for group in groups:
        branches = sorted(group.branches, key=lambda item: int(item.action_id))
        rows.append([float(branch.labels.target_progress_m) for branch in branches])
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (len(groups), assay.ACTION_COUNT) or not np.isfinite(result).all():
        raise MetricValidityError("progress matrix is invalid")
    return result


def geometric_regret_v1(
    groups: Sequence[Any], selected_action_ids: Sequence[int]
) -> dict[str, object]:
    """Compute ``G`` in metres for one arm's per-state selections."""

    progress = progress_matrix_v1(groups)
    if len(selected_action_ids) != len(groups):
        raise MetricValidityError("selection count does not match the panel")
    best = progress.max(axis=1)
    chosen = np.asarray(
        [progress[index, int(action)] for index, action in enumerate(selected_action_ids)],
        dtype=np.float64,
    )
    per_state = best - chosen
    if (per_state < -1.0e-12).any():
        raise MetricValidityError("geometric regret is negative")
    return {
        "geometric_regret_m": float(per_state.mean()),
        "per_state_m": [float(value) for value in per_state],
    }


def rule_based_scores_v1(groups: Sequence[Any], *, rule: str) -> np.ndarray:
    """Score the nine branches under one frozen, non-learned rule.

    ``geometric_endpoint``
        Predicted final distance to target, i.e. privileged endpoint scoring.
    ``bearing``
        Misalignment between the commanded heading and the goal bearing.
    ``hold``
        Always prefer the zero-command branch.
    """

    from lewm.datasets.go2_world_model_counterfactual_pilot_v1 import (
        CANONICAL_ACTION_COMMANDS,
    )

    rows = []
    for group in groups:
        branches = sorted(group.branches, key=lambda item: int(item.action_id))
        goal_x, goal_y = group.relative_target_xy_body_m
        goal_distance = float(np.hypot(goal_x, goal_y))
        goal_bearing = float(np.arctan2(goal_y, goal_x))
        scores = []
        for action, branch in enumerate(branches):
            if rule == "geometric_endpoint":
                scores.append(goal_distance - float(branch.labels.target_progress_m))
            elif rule == "bearing":
                command = CANONICAL_ACTION_COMMANDS[action]
                heading = float(np.arctan2(0.0, command[0])) if command[0] >= 0 else np.pi
                turn = float(command[2])
                predicted = heading + turn
                scores.append(
                    float(
                        abs(
                            np.arctan2(
                                np.sin(predicted - goal_bearing),
                                np.cos(predicted - goal_bearing),
                            )
                        )
                    )
                )
            elif rule == "hold":
                command = CANONICAL_ACTION_COMMANDS[action]
                scores.append(float(abs(command[0]) + abs(command[2])))
            else:
                raise MetricValidityError(f"unknown scoring rule {rule}")
        rows.append(scores)
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (len(groups), assay.ACTION_COUNT) or not np.isfinite(result).all():
        raise MetricValidityError(f"rule {rule} produced invalid scores")
    return result


def divergence_summary_v1(
    groups: Sequence[Any],
    rank_rows: Sequence[Mapping[str, object]],
    geometric: Mapping[str, object],
) -> dict[str, object]:
    """Count states where ``R`` and ``G`` disagree about the chosen branch."""

    progress = progress_matrix_v1(groups)
    ranks = assay.dense_rank_matrix_v1(groups)
    disagreements = 0
    for index in range(len(groups)):
        best_rank = set(np.flatnonzero(ranks[index] == ranks[index].min()).tolist())
        best_progress = set(
            np.flatnonzero(progress[index] == progress[index].max()).tolist()
        )
        if not (best_rank & best_progress):
            disagreements += 1
    return {
        "states": len(groups),
        "states_where_rank_optimum_and_progress_optimum_are_disjoint": disagreements,
        "fraction": float(disagreements) / float(len(groups)),
        "note": (
            "the dense rank orders by (fell, tipped, -progress, path) while G "
            "orders by progress alone, so these states are exactly where the two "
            "one-step metrics can disagree"
        ),
    }


def decide_v1(
    part_b1: Mapping[str, Any], part_b2: Mapping[str, Any]
) -> dict[str, object]:
    """Apply the registered two-link decision rule.

    The chain is valid only if a one-step metric predicts closed-loop utility
    *and* normalized rank regret tracks that metric.
    """

    closed_loop = part_b1["correlation"]
    rank_link = part_b2["correlation"]
    rho_1 = float(closed_loop["spearman"])
    rho_2 = float(rank_link["spearman"])

    closed_loop_passed = (
        rho_1 <= CLOSED_LOOP_LINK_RHO
        and float(closed_loop["ci_upper"]) < CLOSED_LOOP_LINK_BOUND
    )
    rank_passed = (
        rho_2 >= RANK_LINK_RHO and float(rank_link["ci_lower"]) > RANK_LINK_BOUND
    )
    closed_loop_failed = rho_1 > -0.3 or (
        float(closed_loop["ci_lower"]) <= 0.0 <= float(closed_loop["ci_upper"])
    )
    rank_failed = rho_2 < 0.3 or (
        float(rank_link["ci_lower"]) <= 0.0 <= float(rank_link["ci_upper"])
    )

    if closed_loop_passed and rank_passed:
        terminal, reason = VALID_PROXY, (
            "both links hold; normalized rank regret is a usable proxy under a "
            "ceiling-relative threshold"
        )
    elif closed_loop_failed:
        terminal, reason = INVALID_AT_CLOSED_LOOP_LINK, (
            "no one-step endpoint is established as predictive of closed-loop "
            "utility; promote closed-loop progress to the primary endpoint"
        )
    elif closed_loop_passed and rank_failed:
        terminal, reason = INVALID_AT_RANK_LINK, (
            "a one-step metric does predict closed-loop utility, but normalized "
            "rank regret is not that metric; promote geometric first-action "
            "regret in metres"
        )
    else:
        terminal, reason = AMBIGUOUS, (
            "no registered condition holds; report all endpoints and change none"
        )
    return {
        "terminal": terminal,
        "reason": reason,
        "rho_closed_loop_link": rho_1,
        "rho_rank_link": rho_2,
        "closed_loop_link_passed": closed_loop_passed,
        "rank_link_passed": rank_passed,
        "thresholds": {
            "closed_loop_rho": CLOSED_LOOP_LINK_RHO,
            "closed_loop_bound": CLOSED_LOOP_LINK_BOUND,
            "rank_rho": RANK_LINK_RHO,
            "rank_bound": RANK_LINK_BOUND,
        },
        "stops_unchanged": (
            "the section 11 and 13 stops stand regardless of this outcome; this "
            "study changes only which endpoint future preregistrations use"
        ),
    }


def result_identity_v1(result: Mapping[str, object]) -> str:
    payload = {key: value for key, value in result.items() if key != "identity_sha256"}
    return hashlib.sha256(canonical_bytes_v1(payload)).hexdigest()


__all__ = [
    "AMBIGUOUS",
    "BOUND_CLOSED_LOOP_POLICIES",
    "INVALID_AT_CLOSED_LOOP_LINK",
    "INVALID_AT_RANK_LINK",
    "MetricValidityError",
    "SCHEMA",
    "VALID_PROXY",
    "bootstrap_correlation_v1",
    "canonical_bytes_v1",
    "decide_v1",
    "divergence_summary_v1",
    "geometric_regret_v1",
    "part_b1_v1",
    "progress_matrix_v1",
    "result_identity_v1",
    "rule_based_scores_v1",
    "spearman_v1",
]
