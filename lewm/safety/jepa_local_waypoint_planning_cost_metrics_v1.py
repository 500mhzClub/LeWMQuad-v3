"""Pure metrics for JEPA local-waypoint planning-cost qualification V1.

The module is deliberately payload-free.  It accepts already materialised
synthetic or scientific arrays/rows, but performs no corpus discovery, model
loading, rendering, checkpoint access, or filesystem writes.  The scientific
contract lives in a small set of constants and deterministic reductions:

* token-wise L2-normalised, same-position cosine distance averaged over tokens;
* the frozen H3 route-only preference (completion, distance, then heading);
* exact population filters and deterministic cost selections;
* state-, family-, and paired-bootstrap summaries; and
* immutable true-future/predicted gates and classification precedence.

Safety labels are used only to form the prospectively named populations and to
audit selected outcomes.  They never enter the route preference itself.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any

import numpy as np
from scipy import stats


EXPERIMENT_ID = "JEPA_LOCAL_WAYPOINT_PLANNING_COST_QUALIFICATION_V1"

SOURCE_IDS = (
    "TRUE_FUTURE",
    "ONE_STEP_PREDICTED",
    "TWO_STEP_PREDICTED",
)
TRUE_FUTURE, ONE_STEP_PREDICTED, TWO_STEP_PREDICTED = SOURCE_IDS

POPULATION_IDS = (
    "ALL_CANDIDATES",
    "ORACLE_CONTACT_FREE",
    "ORACLE_VIABILITY_ADMISSIBLE",
)
ALL_CANDIDATES, ORACLE_CONTACT_FREE, ORACLE_VIABILITY_ADMISSIBLE = POPULATION_IDS

COMPARATOR_IDS = (
    "KINEMATIC_ROUTE_BASELINE",
    "RANDOM",
    "TRUE_FUTURE_LATENT_COST",
    "ONE_STEP_PREDICTED_LATENT_COST",
    "TWO_STEP_PREDICTED_LATENT_COST",
)
(
    KINEMATIC_ROUTE_BASELINE,
    RANDOM,
    TRUE_FUTURE_LATENT_COST,
    ONE_STEP_PREDICTED_LATENT_COST,
    TWO_STEP_PREDICTED_LATENT_COST,
) = COMPARATOR_IDS

FAMILY_IDS = (
    "large_enclosed_maze",
    "medium_enclosed_maze",
    "small_enclosed_maze",
    "loop_alias_stress",
)
HARD_FAMILY_IDS = ("large_enclosed_maze", "loop_alias_stress")

ROUTE_POPULATION_CLASSES = (
    "TRANSLATIONAL_PROGRESS_AVAILABLE",
    "ALIGNMENT_PROGRESS_AVAILABLE",
    "SAFE_HOLD_OR_ABSTAIN",
    "NO_SAFE_CANDIDATE",
)

DISTANCE_PREFERENCE_MARGIN_M = 0.03
HEADING_PREFERENCE_MARGIN_RAD = math.radians(5.0)
COST_TIE_TOLERANCE = 1.0e-12
NUMERIC_EPSILON = 1.0e-9

TOKEN_COST_DEFINITION = (
    "mean_token_float64(1-dot_float32(l2_normalize_float32("
    "layer_norm_float32(candidate_token)),l2_normalize_float32("
    "layer_norm_float32(goal_token))))"
)
TOKEN_COST_RANGE = (0.0, 2.0)
LAYER_NORM_EPSILON = 1.0e-5
L2_NORMALIZE_FLOOR = 1.0e-12

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 2_026_080_901
BOOTSTRAP_PERCENTILES = (2.5, 97.5)
RANDOM_NAMESPACE = f"{EXPERIMENT_ID}/RANDOM_V1"
STATE_BOOTSTRAP_NAMESPACE = f"{EXPERIMENT_ID}/STATE_BOOTSTRAP_V1"

TRUE_FUTURE_GATE_THRESHOLDS = {
    "pairwise_accuracy_minimum": 0.70,
    "spearman_minimum": 0.60,
    "normalized_regret_maximum": 0.25,
    "best_route_top3_minimum": 0.75,
    "selected_progress_ratio_minimum": 0.80,
    "required_family_count": 4,
}
PREDICTED_GATE_THRESHOLDS = {
    "two_step_pairwise_accuracy_minimum": 0.65,
    "normalized_regret_maximum": 0.30,
    "two_step_to_true_selected_progress_ratio_minimum": 0.75,
    "required_family_count": 4,
}
MATERIAL_IMPROVEMENT_THRESHOLDS = {
    "mean_route_progress_gain_m": 0.05,
    "normalized_regret_reduction": 0.05,
    "hard_family_progress_ratio_gain": 0.10,
    "bootstrap_lower_strictly_greater_than": 0.0,
}

PRIMARY_CLASSIFICATIONS = (
    "RAW_LATENT_GOAL_COST_NO_GO",
    "KINEMATIC_BASELINE_DOMINANT",
    "TWO_STEP_JEPA_PLANNING_COST_SIGNAL",
    "TRUE_FUTURE_COST_SIGNAL_PREDICTOR_PLANNING_NO_GO",
)
SECONDARY_INCREMENTAL_VALUE = "JEPA_INCREMENTAL_ROUTE_VALUE_OVER_KINEMATICS"

NEXT_EXPERIMENTS = (
    "ORACLE_ADMISSIBLE_CLOSED_LOOP_JEPA_MPC_V1",
    "PLAN_AWARE_MONOTONE_JEPA_COST_V1",
    "NON_GREEDY_LOCAL_SUBGOAL_JEPA_PLANNING_V1",
)


class PlanningCostMetricsError(ValueError):
    """Raised for incomplete, ambiguous, or non-finite pure-metric inputs."""


def _finite_scalar(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise PlanningCostMetricsError(f"{name} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise PlanningCostMetricsError(f"{name} must be finite")
    return output


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise PlanningCostMetricsError(f"{name} must be Boolean")
    return bool(value)


def _finite_array(value: Any, *, name: str, minimum_ndim: int = 1) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < minimum_ndim or not array.size:
        raise PlanningCostMetricsError(
            f"{name} must be nonempty with at least {minimum_ndim} dimensions"
        )
    if not np.isfinite(array).all():
        raise PlanningCostMetricsError(f"{name} must contain only finite values")
    return array


def layer_normalize_tokens(
    tokens: Any,
    *,
    epsilon: float = LAYER_NORM_EPSILON,
) -> np.ndarray:
    """Reproduce the frozen ``F.layer_norm`` token transform in float32.

    Persisted latent grids may be FP16 and are never assumed to have already
    passed through ``T.normalise``.  Inputs are first cast to float32, then
    independently normalised over each token's last (feature) dimension using
    population variance and the frozen PyTorch default epsilon.  No learned
    affine parameters are applied.
    """

    array = np.asarray(tokens)
    if array.ndim < 2 or not array.size:
        raise PlanningCostMetricsError(
            "tokens must be nonempty with token and feature dimensions"
        )
    if array.dtype != np.float16:
        raise PlanningCostMetricsError(
            "canonical latent reducer requires persisted float16 token grids"
        )
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise PlanningCostMetricsError("tokens must contain only finite values")
    eps = _finite_scalar(epsilon, name="epsilon")
    if eps <= 0.0:
        raise PlanningCostMetricsError("layer-normalisation epsilon must be positive")
    mean = np.mean(array, axis=-1, keepdims=True, dtype=np.float32)
    centered = np.subtract(array, mean, dtype=np.float32)
    variance = np.mean(
        np.multiply(centered, centered, dtype=np.float32),
        axis=-1,
        keepdims=True,
        dtype=np.float32,
    )
    denominator = np.sqrt(
        np.add(variance, np.float32(eps), dtype=np.float32), dtype=np.float32
    )
    return np.divide(centered, denominator, dtype=np.float32)


def tokenwise_normalized_cosine_costs(
    candidate_tokens: Any,
    goal_tokens: Any,
    *,
    layer_norm_epsilon: float = LAYER_NORM_EPSILON,
    minimum_norm: float = L2_NORMALIZE_FLOOR,
) -> np.ndarray:
    """Return the frozen per-token latent costs with leading broadcasting.

    The last two axes are token and feature.  Both persisted grids are cast to
    float32 and independently layer-normalised before per-token L2
    normalisation.  Dot products and ``1-cosine`` remain float32.  Goal tokens
    may be a single grid broadcast over candidate leading dimensions.  The L2
    denominator uses the exact ``F.normalize`` epsilon clamp, so a zero
    post-layernorm token remains a zero unit vector rather than failing.
    """

    candidate = layer_normalize_tokens(
        candidate_tokens, epsilon=layer_norm_epsilon
    )
    goal = layer_normalize_tokens(goal_tokens, epsilon=layer_norm_epsilon)
    if candidate.shape[-2:] != goal.shape[-2:]:
        raise PlanningCostMetricsError(
            "candidate and goal token/feature dimensions must match exactly"
        )
    floor = _finite_scalar(minimum_norm, name="minimum_norm")
    if floor <= 0.0:
        raise PlanningCostMetricsError("minimum_norm must be strictly positive")
    try:
        candidate, goal = np.broadcast_arrays(candidate, goal)
    except ValueError as exc:
        raise PlanningCostMetricsError(
            "candidate and goal leading dimensions are not broadcast-compatible"
        ) from exc
    candidate_norm = np.sqrt(
        np.sum(
            np.multiply(candidate, candidate, dtype=np.float32),
            axis=-1,
            keepdims=True,
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    goal_norm = np.sqrt(
        np.sum(
            np.multiply(goal, goal, dtype=np.float32),
            axis=-1,
            keepdims=True,
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    candidate_unit = np.divide(
        candidate, np.maximum(candidate_norm, np.float32(floor)), dtype=np.float32
    )
    goal_unit = np.divide(
        goal, np.maximum(goal_norm, np.float32(floor)), dtype=np.float32
    )
    cosine = np.sum(
        np.multiply(candidate_unit, goal_unit, dtype=np.float32),
        axis=-1,
        dtype=np.float32,
    )
    # Floating dot products can escape by a few ulps.  Clipping enforces the
    # mathematical cosine range without changing the registered cost.
    return np.subtract(
        np.float32(1.0),
        np.clip(cosine, np.float32(-1.0), np.float32(1.0)),
        dtype=np.float32,
    )


def tokenwise_normalized_cosine_mean_cost(
    candidate_tokens: Any,
    goal_tokens: Any,
    *,
    layer_norm_epsilon: float = LAYER_NORM_EPSILON,
    minimum_norm: float = L2_NORMALIZE_FLOOR,
) -> float | np.ndarray:
    """Return the registered mean same-position token cosine distance."""

    per_token = tokenwise_normalized_cosine_costs(
        candidate_tokens,
        goal_tokens,
        layer_norm_epsilon=layer_norm_epsilon,
        minimum_norm=minimum_norm,
    )
    result = np.mean(per_token, axis=-1, dtype=np.float64)
    if result.ndim == 0:
        return float(result)
    return result


def trajectory_cost_monotonicity(
    costs_in_temporal_order: Any,
    *,
    tolerance: float = COST_TIE_TOLERANCE,
) -> dict[str, Any]:
    """Measure whether a trajectory's goal cost is non-increasing over time."""

    costs = _finite_array(costs_in_temporal_order, name="costs_in_temporal_order")
    if costs.ndim != 1:
        raise PlanningCostMetricsError("trajectory costs must be one dimensional")
    tol = _finite_scalar(tolerance, name="tolerance")
    if tol < 0.0:
        raise PlanningCostMetricsError("tolerance must be non-negative")
    if len(costs) == 1:
        return {
            "steps": 1,
            "adjacent_comparisons": 0,
            "adjacent_progress_deltas": [],
            "nonincreasing_fraction": None,
            "strict_decrease_fraction": None,
            "violations": 0,
            "violation_count": 0,
            "strict_improvement_count": 0,
            "end_to_end_progress_delta": 0.0,
            "pass_nonincreasing": True,
        }
    # Positive progress means the later latent moved closer to the goal.
    progress_delta = costs[:-1] - costs[1:]
    nonincreasing = progress_delta >= -tol
    strict = progress_delta > tol
    violation_count = int(np.sum(~nonincreasing))
    strict_count = int(np.sum(strict))
    return {
        "steps": int(len(costs)),
        "adjacent_comparisons": int(len(progress_delta)),
        "adjacent_progress_deltas": [float(value) for value in progress_delta],
        "nonincreasing_fraction": float(np.mean(nonincreasing)),
        "strict_decrease_fraction": float(np.mean(strict)),
        "violations": violation_count,
        "violation_count": violation_count,
        "strict_improvement_count": strict_count,
        "end_to_end_progress_delta": float(costs[0] - costs[-1]),
        "pass_nonincreasing": bool(np.all(nonincreasing)),
    }


def _candidate_index(row: Mapping[str, Any]) -> int:
    value = row.get("candidate_index")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise PlanningCostMetricsError("candidate_index must be an integer")
    return int(value)


def _distance_progress(row: Mapping[str, Any]) -> float:
    for key in ("p_d", "p_d_m", "distance_progress_m"):
        if key in row:
            return _finite_scalar(row[key], name=key)
    raise PlanningCostMetricsError("candidate lacks p_d/distance progress")


def _heading_progress(row: Mapping[str, Any]) -> float:
    for key in ("p_theta", "p_theta_rad", "heading_progress_rad"):
        if key in row:
            return _finite_scalar(row[key], name=key)
    raise PlanningCostMetricsError("candidate lacks p_theta/heading progress")


def _completed(row: Mapping[str, Any]) -> bool:
    if "completed" not in row:
        raise PlanningCostMetricsError("candidate lacks completed")
    return _strict_bool(row["completed"], name="completed")


def _oracle_contact(row: Mapping[str, Any]) -> bool:
    for key in ("immediate_contact_h1", "oracle_contact", "contact"):
        if key in row:
            return _strict_bool(row[key], name=key)
    raise PlanningCostMetricsError("candidate lacks oracle_contact")


def _oracle_viable(row: Mapping[str, Any]) -> bool:
    for key in ("oracle_viability_admissible", "viability_admissible"):
        if key in row:
            return _strict_bool(row[key], name=key)
    raise PlanningCostMetricsError("candidate lacks oracle_viability_admissible")


def _successor_viable(row: Mapping[str, Any]) -> bool:
    if "successor_safe_action_count" in row:
        value = row["successor_safe_action_count"]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise PlanningCostMetricsError("successor_safe_action_count must be an integer")
        if not 0 <= int(value) <= 9:
            raise PlanningCostMetricsError(
                "successor_safe_action_count must be in frozen fanout range [0,9]"
            )
        return int(value) > 0
    for key in ("successor_viable", "successor_has_safe_action"):
        if key in row:
            return _strict_bool(row[key], name=key)
    raise PlanningCostMetricsError("candidate lacks independent successor viability")


def _descriptive_contact(row: Mapping[str, Any], horizon: str) -> bool:
    key = f"descriptive_contact_{horizon.lower()}"
    if key not in row:
        raise PlanningCostMetricsError(f"candidate lacks {key}")
    return _strict_bool(row[key], name=key)


def _stuck(row: Mapping[str, Any]) -> bool:
    for key in ("stuck", "oracle_stuck"):
        if key in row:
            return _strict_bool(row[key], name=key)
    raise PlanningCostMetricsError("candidate lacks stuck")


def _nominal_distance_progress(row: Mapping[str, Any]) -> float:
    for key in ("nominal_p_d", "nominal_progress_m", "kinematic_progress_m"):
        if key in row:
            return _finite_scalar(row[key], name=key)
    raise PlanningCostMetricsError("candidate lacks nominal kinematic p_d")


def _nominal_heading_progress(row: Mapping[str, Any]) -> float:
    for key in (
        "nominal_p_theta",
        "nominal_heading_improvement_rad",
        "kinematic_heading_improvement_rad",
    ):
        if key in row:
            return _finite_scalar(row[key], name=key)
    raise PlanningCostMetricsError("candidate lacks nominal kinematic p_theta")


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def kinematic_nominal_outcome(
    post_slew_action_blocks: Any,
    waypoint_xy: Sequence[float],
    *,
    route_heading_rad: float | None = None,
    horizon_blocks: int = 3,
    command_dt_s: float = 0.1,
) -> dict[str, Any]:
    """Integrate the frozen 2-D kinematic comparator through H1--H3.

    Input is exactly three blocks by five ticks by ``(vx, vy, yaw_rate)``.
    Although the registered 10-D predictor action excludes inert ``vy``, the
    preserved post-slew trace retains it and this reducer validates that it is
    zero.  Euler integration processes all 15 100 ms commands, using heading
    at the start of each command.  Only nominal route fields are computed.
    """

    actions = np.asarray(post_slew_action_blocks, dtype=np.float64)
    if actions.shape != (3, 5, 3):
        raise PlanningCostMetricsError(
            "post-slew actions must have frozen shape [3, 5, 3]"
        )
    if isinstance(horizon_blocks, bool) or not isinstance(horizon_blocks, int):
        raise PlanningCostMetricsError("horizon_blocks must be an integer")
    if horizon_blocks != 3:
        raise PlanningCostMetricsError("kinematic baseline requires the first three blocks")
    if not np.isfinite(actions).all():
        raise PlanningCostMetricsError("post-slew actions must be finite")
    if np.any(actions[..., 1] != 0.0):
        raise PlanningCostMetricsError("frozen post-slew lateral velocity vy must be zero")
    waypoint = np.asarray(waypoint_xy, dtype=np.float64)
    if waypoint.shape != (2,) or not np.isfinite(waypoint).all():
        raise PlanningCostMetricsError("waypoint_xy must contain two finite values")
    dt = _finite_scalar(command_dt_s, name="command_dt_s")
    if dt != 0.1:
        raise PlanningCostMetricsError("kinematic baseline command_dt_s is frozen at 0.1")
    heading = (
        math.atan2(float(waypoint[1]), float(waypoint[0]))
        if route_heading_rad is None
        else _finite_scalar(route_heading_rad, name="route_heading_rad")
    )
    x = y = yaw = 0.0
    for vx, vy, yaw_rate in actions.reshape(15, 3):
        x += (math.cos(yaw) * float(vx) - math.sin(yaw) * float(vy)) * dt
        y += (math.sin(yaw) * float(vx) + math.cos(yaw) * float(vy)) * dt
        yaw = _wrap_angle(yaw + float(yaw_rate) * dt)
    start_distance = math.hypot(float(waypoint[0]), float(waypoint[1]))
    end_distance = math.hypot(float(waypoint[0]) - x, float(waypoint[1]) - y)
    p_theta = abs(_wrap_angle(heading)) - abs(_wrap_angle(heading - yaw))
    return {
        "x_m": x,
        "y_m": y,
        "yaw_rad": yaw,
        "nominal_p_d": start_distance - end_distance,
        "nominal_p_theta": p_theta,
        "horizon_blocks": horizon_blocks,
        "integrated_commands": 15,
        "elapsed_s": 15 * dt,
        "command_dt_s": dt,
    }


def kinematic_route_order(candidates: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return positions in the distinct frozen kinematic comparator order.

    At each rank, find the greatest remaining nominal ``p_d``; candidates
    within 0.03 m are eligible; choose greatest nominal ``p_theta`` with *no*
    angular deadband, then ascending candidate index.  Completion and realised
    route/safety fields never enter this comparator.
    """

    rows = list(candidates)
    indices = [_candidate_index(row) for row in rows]
    if len(set(indices)) != len(indices):
        raise PlanningCostMetricsError("candidate indices must be unique")
    remaining = list(range(len(rows)))
    ordered: list[int] = []
    while remaining:
        best_distance = max(_nominal_distance_progress(rows[pos]) for pos in remaining)
        eligible = [
            pos
            for pos in remaining
            if best_distance - _nominal_distance_progress(rows[pos])
            <= DISTANCE_PREFERENCE_MARGIN_M
        ]
        chosen = min(
            eligible,
            key=lambda pos: (
                -_nominal_heading_progress(rows[pos]),
                indices[pos],
            ),
        )
        ordered.append(chosen)
        remaining.remove(chosen)
    return ordered


def kinematic_rank_costs(candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Return zero-based rank costs aligned with input candidate positions."""

    rows = list(candidates)
    output = np.empty(len(rows), dtype=np.float64)
    for rank, position in enumerate(kinematic_route_order(rows)):
        output[position] = float(rank)
    return output


def route_preference(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    distance_margin_m: float = DISTANCE_PREFERENCE_MARGIN_M,
    heading_margin_rad: float = HEADING_PREFERENCE_MARGIN_RAD,
) -> int:
    """Return +1 when ``left`` is route-preferred, -1 for ``right``, else 0.

    This is H3 route-only ordering.  Contact, viability, clearance, and stuck
    values are intentionally ignored: population filtering happens before this
    comparator is applied.
    """

    distance_margin = _finite_scalar(distance_margin_m, name="distance_margin_m")
    heading_margin = _finite_scalar(heading_margin_rad, name="heading_margin_rad")
    if distance_margin < 0.0 or heading_margin < 0.0:
        raise PlanningCostMetricsError("route preference margins must be non-negative")
    left_completed, right_completed = _completed(left), _completed(right)
    if left_completed != right_completed:
        return 1 if left_completed else -1
    distance_delta = _distance_progress(left) - _distance_progress(right)
    if abs(distance_delta) > distance_margin:
        return 1 if distance_delta > 0.0 else -1
    heading_delta = _heading_progress(left) - _heading_progress(right)
    if abs(heading_delta) > heading_margin:
        return 1 if heading_delta > 0.0 else -1
    return 0


def realised_route_order(candidates: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return candidate positions in the frozen realised total order.

    Scientific route ties remain ties for pairwise metrics, but are realised by
    ascending candidate index for selection/top-k accounting.
    """

    rows = list(candidates)
    if not rows:
        return []
    indices = [_candidate_index(row) for row in rows]
    if len(set(indices)) != len(indices):
        raise PlanningCostMetricsError("candidate indices must be unique within state")

    remaining = list(range(len(rows)))
    ordered: list[int] = []
    while remaining:
        preferred_completion = any(_completed(rows[pos]) for pos in remaining)
        completion_eligible = [
            pos for pos in remaining if _completed(rows[pos]) == preferred_completion
        ]
        best_distance = max(
            _distance_progress(rows[pos]) for pos in completion_eligible
        )
        distance_eligible = [
            pos
            for pos in completion_eligible
            if best_distance - _distance_progress(rows[pos])
            <= DISTANCE_PREFERENCE_MARGIN_M
        ]
        best_heading = max(_heading_progress(rows[pos]) for pos in distance_eligible)
        heading_eligible = [
            pos
            for pos in distance_eligible
            if best_heading - _heading_progress(rows[pos])
            <= HEADING_PREFERENCE_MARGIN_RAD
        ]
        chosen = min(heading_eligible, key=lambda pos: indices[pos])
        ordered.append(chosen)
        remaining.remove(chosen)
    return ordered


def margin_borda_utility(candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Return within-state route utility: ``(wins + .5*ties)/(N-1)``."""

    rows = list(candidates)
    count = len(rows)
    if not count:
        return np.empty(0, dtype=np.float64)
    if count == 1:
        return np.ones(1, dtype=np.float64)
    output = np.zeros(count, dtype=np.float64)
    for left in range(count):
        for right in range(left + 1, count):
            preference = route_preference(rows[left], rows[right])
            if preference > 0:
                output[left] += 1.0
            elif preference < 0:
                output[right] += 1.0
            else:
                output[left] += 0.5
                output[right] += 0.5
    return output / (count - 1)


def classify_route_population(candidates: Sequence[Mapping[str, Any]]) -> str:
    """Classify one state using the frozen safe route-intent definitions."""

    rows = list(candidates)
    safe = [row for row in rows if not _oracle_contact(row)]
    if not safe:
        return "NO_SAFE_CANDIDATE"
    if any(
        _completed(row) or _distance_progress(row) > DISTANCE_PREFERENCE_MARGIN_M
        for row in safe
    ):
        return "TRANSLATIONAL_PROGRESS_AVAILABLE"
    if any(_heading_progress(row) > HEADING_PREFERENCE_MARGIN_RAD for row in safe):
        return "ALIGNMENT_PROGRESS_AVAILABLE"
    return "SAFE_HOLD_OR_ABSTAIN"


def filter_population(
    candidates: Sequence[Mapping[str, Any]], population_id: str
) -> list[Mapping[str, Any]]:
    """Apply one exact population filter without changing candidate order."""

    if population_id not in POPULATION_IDS:
        raise PlanningCostMetricsError(f"unknown population {population_id!r}")
    rows = list(candidates)
    for row in rows:
        expected_admissible = (not _oracle_contact(row)) and _successor_viable(row)
        if _oracle_viable(row) != expected_admissible:
            raise PlanningCostMetricsError(
                "oracle_viability_admissible must equal contact_free_h1 AND successor_viable"
            )
    if any(_oracle_viable(row) and _oracle_contact(row) for row in rows):
        raise PlanningCostMetricsError(
            "oracle-viability population must be nested inside contact-free population"
        )
    if population_id == ALL_CANDIDATES:
        return rows
    if population_id == ORACLE_CONTACT_FREE:
        return [row for row in rows if not _oracle_contact(row)]
    return [row for row in rows if _oracle_viable(row)]


def deterministic_random_order(
    state_id: str,
    candidate_indices: Sequence[int],
    *,
    seed: int = BOOTSTRAP_SEED,
) -> list[int]:
    """Return the frozen SHA-256 random baseline order.

    Bytes are exactly namespace UTF-8, NUL, unsigned seed uint64-BE, NUL,
    state ID UTF-8, NUL, candidate index uint32-BE.  Sorting is ascending
    digest then candidate index.
    """

    if not isinstance(state_id, str) or not state_id:
        raise PlanningCostMetricsError("state_id must be a nonempty string")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**64
    ):
        raise PlanningCostMetricsError("seed must be an unsigned uint64")
    values = []
    for raw in candidate_indices:
        if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
            raise PlanningCostMetricsError("candidate indices must be integers")
        value = int(raw)
        if not 0 <= value < 2**32:
            raise PlanningCostMetricsError("candidate index must be an unsigned uint32")
        values.append(value)
    if len(set(values)) != len(values):
        raise PlanningCostMetricsError("candidate indices must be unique")
    prefix = (
        RANDOM_NAMESPACE.encode("utf-8")
        + b"\x00"
        + seed.to_bytes(8, "big", signed=False)
        + b"\x00"
        + state_id.encode("utf-8")
        + b"\x00"
    )
    return sorted(
        values,
        key=lambda value: (
            hashlib.sha256(prefix + value.to_bytes(4, "big", signed=False)).digest(),
            value,
        ),
    )


def _cost_order(
    costs: np.ndarray,
    candidate_indices: Sequence[int],
    *,
    tolerance: float = COST_TIE_TOLERANCE,
) -> list[int]:
    """Order costs with the frozen tolerance tie and candidate-index rule.

    Repeatedly form the tie set within ``tolerance`` of the lowest remaining
    cost, then select the lowest frozen candidate index from that set.  This
    avoids a non-transitive comparison sort while ensuring that a difference
    at or below the registered tolerance never determines selection or rank.
    """

    tol = _finite_scalar(tolerance, name="cost_order_tolerance")
    if tol < 0.0:
        raise PlanningCostMetricsError("cost-order tolerance must be non-negative")
    if len(costs) != len(candidate_indices):
        raise PlanningCostMetricsError("costs and candidate indices must align")
    remaining = list(range(len(costs)))
    ordered: list[int] = []
    while remaining:
        minimum = min(float(costs[position]) for position in remaining)
        tied = [
            position
            for position in remaining
            if float(costs[position]) - minimum <= tol
        ]
        chosen = min(tied, key=lambda position: candidate_indices[position])
        ordered.append(chosen)
        remaining.remove(chosen)
    return ordered


def _correlation(value: Any) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def route_ordering_metrics(
    candidates: Sequence[Mapping[str, Any]],
    costs: Any,
    *,
    cost_tie_tolerance: float = COST_TIE_TOLERANCE,
) -> dict[str, Any]:
    """Evaluate route ordering for one already-filtered state population."""

    rows = list(candidates)
    numeric = np.asarray(costs, dtype=np.float64)
    if numeric.shape != (len(rows),) or (len(rows) and not np.isfinite(numeric).all()):
        raise PlanningCostMetricsError("costs must be one finite value per candidate")
    tolerance = _finite_scalar(cost_tie_tolerance, name="cost_tie_tolerance")
    if tolerance < 0.0:
        raise PlanningCostMetricsError("cost tie tolerance must be non-negative")
    indices = [_candidate_index(row) for row in rows]
    if len(set(indices)) != len(indices):
        raise PlanningCostMetricsError("candidate indices must be unique")
    if not rows:
        return {
            "candidates": 0,
            "ordered_pairs": 0,
            "pairwise_correct_credit": 0.0,
            "pairwise_accuracy": None,
            "spearman_rho": None,
            "kendall_tau_b": None,
            "oracle_best_candidate_index": None,
            "selected_candidate_index": None,
            "oracle_best_cost_rank": None,
            "best_route_top1": None,
            "best_route_top3": None,
            "mean_reciprocal_rank": None,
            "mean_best_route_rank": None,
            "cost_minimum": None,
            "cost_maximum": None,
            "cost_spread": None,
            "cost_pair_count": 0,
            "cost_tie_count": 0,
            "cost_tie_rate": None,
            "cost_tie_pair_count": 0,
            "cost_tie_pair_rate": None,
            "selected_combined_utility": None,
            "ranked_candidate_indices": [],
            "best_route_rank": None,
            "best_route_reciprocal_rank": None,
            "ordered_pair_count": 0,
            "pairwise_denominator_ordered_pairs": 0,
            "oracle_realised_order": [],
            "cost_order": [],
            "margin_borda_utility": {},
        }
    correct_credit = 0.0
    ordered_pairs = 0
    cost_pair_count = 0
    cost_tie_count = 0
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            cost_pair_count += 1
            delta = float(numeric[left] - numeric[right])
            if abs(delta) <= tolerance:
                cost_tie_count += 1
            preference = route_preference(rows[left], rows[right])
            if preference == 0:
                continue
            ordered_pairs += 1
            if abs(delta) <= tolerance:
                correct_credit += 0.5
            elif (preference > 0 and delta < 0.0) or (preference < 0 and delta > 0.0):
                correct_credit += 1.0
    distance = np.asarray([_distance_progress(row) for row in rows], np.float64)
    if (
        len(rows) >= 2
        and np.unique(numeric).size >= 2
        and np.unique(distance).size >= 2
    ):
        spearman = stats.spearmanr(-numeric, distance).statistic
        kendall = stats.kendalltau(-numeric, distance, variant="b").statistic
        spearman_value = _correlation(spearman)
        kendall_value = _correlation(kendall)
    else:
        spearman_value = None
        kendall_value = None
    oracle_positions = realised_route_order(rows)
    cost_positions = _cost_order(numeric, indices, tolerance=tolerance)
    oracle_best_position = oracle_positions[0]
    oracle_best_candidate = indices[oracle_best_position]
    selected_candidate = indices[cost_positions[0]]
    rank = 1 + cost_positions.index(oracle_best_position)
    utilities = margin_borda_utility(rows)
    selected_position = cost_positions[0]
    cost_minimum = float(np.min(numeric))
    cost_maximum = float(np.max(numeric))
    return {
        "candidates": len(rows),
        "ordered_pairs": ordered_pairs,
        "pairwise_correct_credit": correct_credit,
        "pairwise_accuracy": (
            correct_credit / ordered_pairs if ordered_pairs else None
        ),
        "spearman_rho": spearman_value,
        "kendall_tau_b": kendall_value,
        "oracle_best_candidate_index": oracle_best_candidate,
        "selected_candidate_index": selected_candidate,
        "oracle_best_cost_rank": rank,
        "mean_best_route_rank": float(rank),
        "best_route_top1": rank == 1,
        "best_route_top3": rank <= 3,
        "mean_reciprocal_rank": 1.0 / rank,
        "cost_minimum": cost_minimum,
        "cost_maximum": cost_maximum,
        "cost_spread": cost_maximum - cost_minimum,
        "cost_pair_count": cost_pair_count,
        "cost_tie_count": cost_tie_count,
        "cost_tie_rate": (
            cost_tie_count / cost_pair_count if cost_pair_count else None
        ),
        "cost_tie_pair_count": cost_tie_count,
        "cost_tie_pair_rate": (
            cost_tie_count / cost_pair_count if cost_pair_count else None
        ),
        "selected_combined_utility": float(utilities[selected_position]),
        "oracle_realised_order": [indices[position] for position in oracle_positions],
        "cost_order": [indices[position] for position in cost_positions],
        "ranked_candidate_indices": [indices[position] for position in cost_positions],
        "best_route_rank": rank,
        "best_route_reciprocal_rank": 1.0 / rank,
        "ordered_pair_count": ordered_pairs,
        "pairwise_denominator_ordered_pairs": ordered_pairs,
        "margin_borda_utility": {
            str(indices[position]): float(utilities[position])
            for position in range(len(rows))
        },
    }


def evaluate_state_population(
    candidates: Sequence[Mapping[str, Any]],
    costs: Any,
    *,
    state_id: str,
    family: str,
    source_id: str,
    population_id: str,
    role: str | None = None,
) -> dict[str, Any]:
    """Reduce one state's candidate population through ranking and selection."""

    if source_id not in SOURCE_IDS and source_id not in COMPARATOR_IDS:
        raise PlanningCostMetricsError(f"unknown source/comparator {source_id!r}")
    if not isinstance(state_id, str) or not state_id:
        raise PlanningCostMetricsError("state_id must be nonempty")
    if not isinstance(family, str) or not family:
        raise PlanningCostMetricsError("family must be nonempty")
    all_rows = list(candidates)
    if role is None:
        roles = {str(row.get("role", "")) for row in all_rows}
        if len(roles) != 1 or "" in roles:
            raise PlanningCostMetricsError(
                "state candidates must bind exactly one nonempty role"
            )
        role = next(iter(roles))
    if not isinstance(role, str) or not role:
        raise PlanningCostMetricsError("role must be nonempty")
    all_costs = np.asarray(costs, dtype=np.float64)
    if all_costs.shape != (len(all_rows),) or (
        len(all_rows) and not np.isfinite(all_costs).all()
    ):
        raise PlanningCostMetricsError("costs must align one-to-one with candidates")
    filtered = filter_population(all_rows, population_id)
    positions_by_identity = {id(row): position for position, row in enumerate(all_rows)}
    filtered_costs = np.asarray(
        [all_costs[positions_by_identity[id(row)]] for row in filtered], np.float64
    )
    ordering = route_ordering_metrics(filtered, filtered_costs)
    population_class = classify_route_population(all_rows)
    selected_index = ordering["selected_candidate_index"]
    oracle_best_index = ordering["oracle_best_candidate_index"]
    filtered_by_index = {_candidate_index(row): row for row in filtered}
    if selected_index is None:
        selected = None
        selected_progress = 0.0
        selected_heading = 0.0
        selected_completed = False
        selected_contact = False
        selected_contact_h2 = False
        selected_contact_h3 = False
        selected_nonviable = False
        selected_stuck = False
        selected_utility = None
        normalized_regret = None
    else:
        selected = filtered_by_index[int(selected_index)]
        selected_progress = _distance_progress(selected)
        selected_heading = _heading_progress(selected)
        selected_completed = _completed(selected)
        selected_contact = _oracle_contact(selected)
        selected_contact_h2 = _descriptive_contact(selected, "H2")
        selected_contact_h3 = _descriptive_contact(selected, "H3")
        selected_nonviable = not _successor_viable(selected)
        selected_stuck = _stuck(selected)
        selected_utility = float(ordering["selected_combined_utility"])
        progress_values = np.asarray(
            [_distance_progress(row) for row in filtered], np.float64
        )
        progress_range = float(np.max(progress_values) - np.min(progress_values))
        best = filtered_by_index[int(oracle_best_index)]
        best_progress = _distance_progress(best)
        if progress_range <= COST_TIE_TOLERANCE:
            if selected_progress == best_progress:
                normalized_regret = 0.0
            else:
                raise PlanningCostMetricsError(
                    "p_d range at or below 1e-12 requires selected p_d to equal "
                    "route-best p_d"
                )
        else:
            normalized_regret = (
                best_progress - selected_progress
            ) / progress_range
    oracle_best_progress = (
        0.0
        if oracle_best_index is None
        else _distance_progress(filtered_by_index[int(oracle_best_index)])
    )
    abstained = selected is None
    return {
        "schema": "jepa_local_waypoint_state_population_metrics_v1",
        "state_id": state_id,
        "family": family,
        "role": role,
        "source_id": source_id,
        "population_id": population_id,
        "route_population_class": population_class,
        **ordering,
        "abstained": abstained,
        "abstention": abstained,
        # Ranking is total over every nonempty population.  Therefore an
        # abstention is exactly an empty-population outcome; it is never
        # reclassified from progress present elsewhere in the candidate bank.
        "correct_abstention": abstained,
        "false_abstention": False,
        "empty_population_abstention": abstained,
        "no_oracle_contact_free_action": (
            abstained and population_id == ORACLE_CONTACT_FREE
        ),
        "no_oracle_viability_admissible_action": (
            abstained and population_id == ORACLE_VIABILITY_ADMISSIBLE
        ),
        "selected_completed": selected_completed,
        "completion": selected_completed,
        "completed": selected_completed,
        "selected_route_progress_m": selected_progress,
        "selected_progress_m": selected_progress,
        "selected_p_d": selected_progress,
        "selected_heading_progress_rad": selected_heading,
        "selected_heading_improvement_rad": selected_heading,
        "selected_p_theta": selected_heading,
        "selected_oracle_contact": selected_contact,
        "selected_contact": selected_contact,
        "selected_immediate_contact": selected_contact,
        "selected_immediate_contact_h1": selected_contact,
        "selected_descriptive_contact_h2": selected_contact_h2,
        "selected_descriptive_contact_h3": selected_contact_h3,
        "selected_oracle_nonviable": selected_nonviable,
        "selected_nonviable": selected_nonviable,
        "selected_successor_nonviable": selected_nonviable,
        "selected_nonviable_successor": selected_nonviable,
        "selected_stuck": selected_stuck,
        "selected_combined_utility": selected_utility,
        "oracle_best_route_progress_m": oracle_best_progress,
        "oracle_best_progress_m": oracle_best_progress,
        "normalized_regret": normalized_regret,
        "normalised_regret": normalized_regret,
    }


def _optional_mean(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(np.mean(present)) if present else None


def aggregate_state_metrics(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate complete per-state reductions without candidate-row weighting."""

    rows = list(states)
    if not rows:
        return {
            "states": 0,
            "candidates": 0,
            "evaluable_nonabstaining_states": 0,
            "states_with_ordered_pairs": 0,
            "ordered_pairs": 0,
            "pairwise_correct_credit": 0.0,
            "pairwise_accuracy": None,
            "spearman_rho": None,
            "spearman": None,
            "spearman_mean_per_state": None,
            "kendall_tau_b": None,
            "kendall": None,
            "kendall_mean_per_state": None,
            "best_route_top1_rate": None,
            "best_route_top1": None,
            "best_route_top3_rate": None,
            "best_route_top3": None,
            "mean_reciprocal_rank": None,
            "mrr": None,
            "mean_best_route_rank": None,
            "cost_spread": None,
            "cost_tie_count": 0,
            "cost_ties": 0,
            "cost_tie_pair_count": 0,
            "cost_pair_count": 0,
            "cost_tie_rate": None,
            "cost_tie_pair_rate": None,
            "normalized_regret": None,
            "normalised_regret": None,
            "selected_route_progress_m_sum": 0.0,
            "aggregate_selected_route_progress_m": 0.0,
            "selected_progress_sum_m": 0.0,
            "selected_route_progress_sum_m": 0.0,
            "selected_progress_mean_m": 0.0,
            "selected_progress_m": 0.0,
            "selected_heading_improvement_rad": 0.0,
            "selected_combined_utility": None,
            "oracle_best_route_progress_m_sum": 0.0,
            "oracle_best_progress_sum_m": 0.0,
            "oracle_best_progress_mean_m": 0.0,
            "selected_progress_ratio": 0.0,
            "selected_progress_fraction_of_oracle_best": 0.0,
            "abstentions": 0,
            "abstention": 0,
            "selected_completions": 0,
            "completion": 0,
            "selected_identity_counts": {},
            "selected_oracle_contacts": 0,
            "selected_contacts": 0,
            "selected_immediate_contacts_h1": 0,
            "selected_descriptive_contacts_h2": 0,
            "selected_descriptive_contacts_h3": 0,
            "selected_oracle_nonviable": 0,
            "selected_nonviable": 0,
            "selected_nonviable_successors": 0,
            "selected_stuck": 0,
            "complete_family_collapse": True,
        }
    state_ids = [str(row["state_id"]) for row in rows]
    if len(set(state_ids)) != len(state_ids):
        raise PlanningCostMetricsError("per-state aggregate contains duplicate state IDs")
    ordered_pairs = sum(int(row["ordered_pairs"]) for row in rows)
    correct_credit = sum(float(row["pairwise_correct_credit"]) for row in rows)
    nonempty = [row for row in rows if not bool(row["abstained"])]
    top_rows = [row for row in rows if row["best_route_top3"] is not None]
    selected_sum = float(sum(float(row["selected_route_progress_m"]) for row in rows))
    selected_heading_sum = float(
        sum(float(row["selected_heading_progress_rad"]) for row in rows)
    )
    best_sum = float(sum(float(row["oracle_best_route_progress_m"]) for row in rows))
    pairwise_accuracy = correct_credit / ordered_pairs if ordered_pairs else None
    top3_rate = (
        float(np.mean([bool(row["best_route_top3"]) for row in top_rows]))
        if top_rows
        else None
    )
    positive_signal = bool(
        (pairwise_accuracy is not None and pairwise_accuracy > 0.5)
        or (top3_rate is not None and top3_rate > 0.0)
        or selected_sum > 0.0
    )
    collapsed = not nonempty or ordered_pairs == 0 or not positive_signal
    spearman = _optional_mean([row["spearman_rho"] for row in rows])
    kendall = _optional_mean([row["kendall_tau_b"] for row in rows])
    top1 = _optional_mean(
        [
            None if row["best_route_top1"] is None else float(row["best_route_top1"])
            for row in rows
        ]
    )
    mrr = _optional_mean([row["mean_reciprocal_rank"] for row in rows])
    mean_rank = _optional_mean([row["mean_best_route_rank"] for row in rows])
    cost_spread = _optional_mean([row["cost_spread"] for row in rows])
    cost_pair_count = sum(int(row["cost_pair_count"]) for row in rows)
    cost_tie_count = sum(int(row["cost_tie_count"]) for row in rows)
    regret = _optional_mean([row["normalized_regret"] for row in rows])
    selected_utility = _optional_mean(
        [row["selected_combined_utility"] for row in rows]
    )
    contacts = sum(bool(row["selected_oracle_contact"]) for row in rows)
    contacts_h2 = sum(bool(row["selected_descriptive_contact_h2"]) for row in rows)
    contacts_h3 = sum(bool(row["selected_descriptive_contact_h3"]) for row in rows)
    nonviable = sum(bool(row["selected_oracle_nonviable"]) for row in rows)
    stuck = sum(bool(row["selected_stuck"]) for row in rows)
    completions = sum(bool(row["selected_completed"]) for row in rows)
    abstentions = sum(bool(row["abstained"]) for row in rows)
    identity_counts: dict[str, int] = defaultdict(int)
    for row in nonempty:
        identity_counts[str(int(row["selected_candidate_index"]))] += 1
    return {
        "states": len(rows),
        "candidates": sum(int(row["candidates"]) for row in rows),
        "evaluable_nonabstaining_states": len(nonempty),
        "states_with_ordered_pairs": sum(int(row["ordered_pairs"]) > 0 for row in rows),
        "ordered_pairs": ordered_pairs,
        "pairwise_correct_credit": correct_credit,
        "pairwise_accuracy": pairwise_accuracy,
        "spearman_rho": spearman,
        "spearman": spearman,
        "spearman_mean_per_state": spearman,
        "kendall_tau_b": kendall,
        "kendall": kendall,
        "kendall_mean_per_state": kendall,
        "best_route_top1_rate": top1,
        "best_route_top1": top1,
        "best_route_top3_rate": top3_rate,
        "best_route_top3": top3_rate,
        "mean_reciprocal_rank": mrr,
        "mrr": mrr,
        "mean_best_route_rank": mean_rank,
        "cost_spread": cost_spread,
        "cost_tie_count": cost_tie_count,
        "cost_ties": cost_tie_count,
        "cost_tie_pair_count": cost_tie_count,
        "cost_pair_count": cost_pair_count,
        "cost_tie_rate": cost_tie_count / cost_pair_count if cost_pair_count else None,
        "cost_tie_pair_rate": cost_tie_count / cost_pair_count if cost_pair_count else None,
        "normalized_regret": regret,
        "normalised_regret": regret,
        "selected_route_progress_m_sum": selected_sum,
        "aggregate_selected_route_progress_m": selected_sum,
        "selected_route_progress_m_mean": selected_sum / len(rows),
        "selected_progress_sum_m": selected_sum,
        "selected_route_progress_sum_m": selected_sum,
        "selected_progress_mean_m": selected_sum / len(rows),
        "selected_progress_m": selected_sum,
        "selected_heading_improvement_rad": selected_heading_sum,
        "selected_combined_utility": selected_utility,
        "oracle_best_route_progress_m_sum": best_sum,
        "oracle_best_route_progress_m_mean": best_sum / len(rows),
        "oracle_best_progress_sum_m": best_sum,
        "oracle_best_progress_mean_m": best_sum / len(rows),
        "selected_progress_ratio": selected_sum / max(abs(best_sum), NUMERIC_EPSILON),
        "selected_progress_fraction_of_oracle_best": selected_sum
        / max(abs(best_sum), NUMERIC_EPSILON),
        "abstentions": abstentions,
        "abstention": abstentions,
        "correct_abstentions": sum(bool(row["correct_abstention"]) for row in rows),
        "false_abstentions": sum(bool(row["false_abstention"]) for row in rows),
        "selected_completions": completions,
        "completion": completions,
        "selected_identity_counts": dict(sorted(identity_counts.items())),
        "selected_oracle_contacts": contacts,
        "selected_contacts": contacts,
        "selected_immediate_contacts_h1": contacts,
        "selected_descriptive_contacts_h2": contacts_h2,
        "selected_descriptive_contacts_h3": contacts_h3,
        "selected_oracle_nonviable": nonviable,
        "selected_nonviable": nonviable,
        "selected_nonviable_successors": nonviable,
        "selected_stuck": stuck,
        "complete_family_collapse": collapsed,
    }


def summarize_state_families(
    states: Sequence[Mapping[str, Any]],
    *,
    expected_families: Sequence[str] = FAMILY_IDS,
) -> dict[str, Any]:
    """Report pooled, per-family, per-role, and collapse summaries."""

    rows = list(states)
    families = tuple(expected_families)
    if not families or len(set(families)) != len(families):
        raise PlanningCostMetricsError("expected_families must be nonempty and unique")
    observed = {str(row["family"]) for row in rows}
    if observed - set(families):
        raise PlanningCostMetricsError(
            f"unexpected families {sorted(observed - set(families))}"
        )
    per_family = {
        family: aggregate_state_metrics(
            [row for row in rows if str(row["family"]) == family]
        )
        for family in families
    }
    collapsed = [
        family
        for family, summary in per_family.items()
        if summary["complete_family_collapse"]
    ]
    roles = sorted({str(row.get("role", "")) for row in rows})
    if "" in roles:
        raise PlanningCostMetricsError("each state metric must bind a nonempty role")
    per_role = {
        role: aggregate_state_metrics(
            [row for row in rows if str(row["role"]) == role]
        )
        for role in roles
    }
    return {
        "aggregate": aggregate_state_metrics(rows),
        "per_family": per_family,
        "per_role": per_role,
        "expected_families": list(families),
        "collapsed_families": collapsed,
        "no_family_complete_collapse": not collapsed,
    }


def summarize_source(
    candidates_by_state: Mapping[str, Sequence[Mapping[str, Any]]],
    costs_by_state: Mapping[str, Any],
    *,
    source_id: str,
    expected_families: Sequence[str] = FAMILY_IDS,
) -> dict[str, Any]:
    """Evaluate all three frozen populations for one source/comparator."""

    candidate_state_order = list(candidates_by_state)
    cost_state_order = list(costs_by_state)
    if candidate_state_order != cost_state_order:
        raise PlanningCostMetricsError(
            "candidate/cost state identities and frozen manifest order must match exactly"
        )
    population_output: dict[str, Any] = {}
    for population_id in POPULATION_IDS:
        per_state = []
        # Mapping insertion order is the caller-validated frozen state-manifest
        # order.  Never lexicographically sort IDs such as purpose-2/purpose-10:
        # that would silently alter the registered state bootstrap draw.
        for state_id in candidate_state_order:
            candidates = list(candidates_by_state[state_id])
            families = {str(row.get("family", "")) for row in candidates}
            if len(families) != 1 or "" in families:
                raise PlanningCostMetricsError(
                    f"state {state_id} must have exactly one nonempty family"
                )
            roles = {str(row.get("role", "")) for row in candidates}
            if len(roles) != 1 or "" in roles:
                raise PlanningCostMetricsError(
                    f"state {state_id} must have exactly one nonempty role"
                )
            per_state.append(
                evaluate_state_population(
                    candidates,
                    costs_by_state[state_id],
                    state_id=state_id,
                    family=next(iter(families)),
                    source_id=source_id,
                    population_id=population_id,
                    role=next(iter(roles)),
                )
            )
        summary = summarize_state_families(
            per_state, expected_families=expected_families
        )
        population_output[population_id] = {"per_state": per_state, **summary}
    return {
        "schema": "jepa_local_waypoint_planning_cost_source_metrics_v1",
        "source_id": source_id,
        "populations": population_output,
    }


def _continuous_order_diagnostic(
    predictor: np.ndarray, outcome: np.ndarray
) -> dict[str, Any]:
    if predictor.shape != outcome.shape or predictor.ndim != 1:
        raise PlanningCostMetricsError("diagnostic vectors must be aligned and 1-D")
    if not np.isfinite(predictor).all() or not np.isfinite(outcome).all():
        raise PlanningCostMetricsError("diagnostic vectors must be finite")
    if (
        len(predictor) >= 2
        and np.unique(predictor).size >= 2
        and np.unique(outcome).size >= 2
    ):
        spearman = _correlation(stats.spearmanr(predictor, outcome).statistic)
        kendall = _correlation(
            stats.kendalltau(predictor, outcome, variant="b").statistic
        )
    else:
        spearman = kendall = None
    ordered_pairs = 0
    credit = 0.0
    for left in range(len(predictor)):
        for right in range(left + 1, len(predictor)):
            outcome_delta = float(outcome[left] - outcome[right])
            if outcome_delta == 0.0:
                continue
            ordered_pairs += 1
            predictor_delta = float(predictor[left] - predictor[right])
            if abs(predictor_delta) <= COST_TIE_TOLERANCE:
                credit += 0.5
            elif (predictor_delta > 0.0) == (outcome_delta > 0.0):
                credit += 1.0
    return {
        "spearman_rho": spearman,
        "kendall_tau_b": kendall,
        "ordered_pairs": ordered_pairs,
        "pairwise_correct_credit": credit,
        "pairwise_accuracy": credit / ordered_pairs if ordered_pairs else None,
    }


def _horizon_route_row(row: Mapping[str, Any], horizon: str) -> dict[str, Any]:
    suffix = horizon.lower()
    required = {
        "p_d": f"p_d_{suffix}",
        "p_theta": f"p_theta_{suffix}",
        "completed": f"completed_{suffix}",
    }
    missing = [key for key in required.values() if key not in row]
    if missing:
        raise PlanningCostMetricsError(
            f"candidate lacks horizon-aligned route fields {missing}"
        )
    return {
        "candidate_index": _candidate_index(row),
        "p_d": _finite_scalar(row[required["p_d"]], name=required["p_d"]),
        "p_theta": _finite_scalar(
            row[required["p_theta"]], name=required["p_theta"]
        ),
        "completed": _strict_bool(
            row[required["completed"]], name=required["completed"]
        ),
    }


def _diagnostic_group(rows: Sequence[Mapping[str, Any]], horizon: str) -> dict[str, Any]:
    cost_key = f"cost_{horizon.lower()}"
    delta = np.asarray(
        [
            _finite_scalar(row["cost_current"], name="cost_current")
            - _finite_scalar(row[cost_key], name=cost_key)
            for row in rows
        ],
        dtype=np.float64,
    )
    horizon_rows = [_horizon_route_row(row, horizon) for row in rows]
    distance = np.asarray([_distance_progress(row) for row in horizon_rows], np.float64)
    heading = np.asarray([_heading_progress(row) for row in horizon_rows], np.float64)
    utility = np.asarray(
        [
            _finite_scalar(
                row[f"combined_margin_borda_utility_{horizon.lower()}"],
                name=f"combined_margin_borda_utility_{horizon.lower()}",
            )
            for row in rows
        ],
        np.float64,
    )
    return {
        "candidates": len(rows),
        "latent_goal_distance_change_definition": f"cost_current-cost_{horizon.lower()}",
        "latent_goal_distance_change_mean": (
            float(np.mean(delta)) if len(delta) else None
        ),
        "against_realized_distance_progress": _continuous_order_diagnostic(
            delta, distance
        ),
        "against_realized_heading_improvement": _continuous_order_diagnostic(
            delta, heading
        ),
        "against_combined_margin_borda_route_intent": _continuous_order_diagnostic(
            delta, utility
        ),
    }


def _monotonic_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "trajectories": 0,
            "current_to_h1_nonincrease_fraction": None,
            "h1_to_h2_nonincrease_fraction": None,
            "h2_to_h3_nonincrease_fraction": None,
            "current_to_h1_strict_decrease_fraction": None,
            "h1_to_h2_strict_decrease_fraction": None,
            "h2_to_h3_strict_decrease_fraction": None,
            "current_to_h1_decrease_fraction": None,
            "h1_to_h2_decrease_fraction": None,
            "h2_to_h3_decrease_fraction": None,
            "current_to_h3_nonincrease_fraction": None,
            "current_to_h3_strict_decrease_fraction": None,
            "current_to_h3_decrease_fraction": None,
            "overall_monotonic_trajectory_fraction": None,
            "overall_monotonic_nonincrease_fraction": None,
            "overall_strict_decrease_trajectory_fraction": None,
        }
    nonincrease_flags = []
    strict_flags = []
    for row in rows:
        values = np.asarray(
            [row["cost_current"], row["cost_h1"], row["cost_h2"], row["cost_h3"]],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise PlanningCostMetricsError("monotonic diagnostic costs must be finite")
        delta = values[:-1] - values[1:]
        nonincrease_flags.append(
            [bool(value) for value in delta >= -COST_TIE_TOLERANCE]
        )
        strict_flags.append([bool(value) for value in delta > COST_TIE_TOLERANCE])
    nonincrease = np.asarray(nonincrease_flags, dtype=bool)
    strict = np.asarray(strict_flags, dtype=bool)
    endpoint_delta = np.asarray(
        [float(row["cost_current"]) - float(row["cost_h3"]) for row in rows],
        dtype=np.float64,
    )
    return {
        "trajectories": len(rows),
        "current_to_h1_nonincrease_fraction": float(np.mean(nonincrease[:, 0])),
        "h1_to_h2_nonincrease_fraction": float(np.mean(nonincrease[:, 1])),
        "h2_to_h3_nonincrease_fraction": float(np.mean(nonincrease[:, 2])),
        "current_to_h1_strict_decrease_fraction": float(np.mean(strict[:, 0])),
        "h1_to_h2_strict_decrease_fraction": float(np.mean(strict[:, 1])),
        "h2_to_h3_strict_decrease_fraction": float(np.mean(strict[:, 2])),
        "current_to_h1_decrease_fraction": float(np.mean(strict[:, 0])),
        "h1_to_h2_decrease_fraction": float(np.mean(strict[:, 1])),
        "h2_to_h3_decrease_fraction": float(np.mean(strict[:, 2])),
        "current_to_h3_nonincrease_fraction": float(
            np.mean(endpoint_delta >= -COST_TIE_TOLERANCE)
        ),
        "current_to_h3_strict_decrease_fraction": float(
            np.mean(endpoint_delta > COST_TIE_TOLERANCE)
        ),
        "current_to_h3_decrease_fraction": float(
            np.mean(endpoint_delta > COST_TIE_TOLERANCE)
        ),
        "overall_monotonic_trajectory_fraction": float(
            np.mean(np.all(nonincrease, axis=1))
        ),
        "overall_monotonic_nonincrease_fraction": float(
            np.mean(np.all(nonincrease, axis=1))
        ),
        "overall_strict_decrease_trajectory_fraction": float(
            np.mean(np.all(strict, axis=1))
        ),
    }


def latent_progress_diagnostics(
    rows: Sequence[Mapping[str, Any]], *, source_id: str
) -> dict[str, Any]:
    """Reduce the registered no-gate latent-progress/monotonic diagnostics.

    Rows represent candidate trajectories and must contain CURRENT/H1/H2/H3
    costs plus realised H3 route fields.  Combined margin-Borda utility is
    derived within each state, never supplied or tuned by the caller.
    """

    if source_id not in SOURCE_IDS:
        raise PlanningCostMetricsError("diagnostics require a frozen latent source")
    copied = [dict(row) for row in rows]
    grouped_states: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in copied:
        state_id = str(row.get("state_id", ""))
        family = str(row.get("family", ""))
        role = str(row.get("role", ""))
        if not state_id or not family or not role:
            raise PlanningCostMetricsError(
                "diagnostic row must bind state_id, family, and role"
            )
        grouped_states[state_id].append(row)
    for state_rows in grouped_states.values():
        indices = [_candidate_index(row) for row in state_rows]
        if len(set(indices)) != len(indices):
            raise PlanningCostMetricsError("diagnostic state has duplicate candidates")
        for horizon in ("H1", "H2", "H3"):
            utilities = margin_borda_utility(
                [_horizon_route_row(row, horizon) for row in state_rows]
            )
            for row, utility in zip(state_rows, utilities):
                row[f"combined_margin_borda_utility_{horizon.lower()}"] = float(
                    utility
                )
    flattened = [
        row
        for state_id in grouped_states
        for row in sorted(grouped_states[state_id], key=_candidate_index)
    ]
    families = sorted({str(row["family"]) for row in flattened})
    roles = sorted({str(row["role"]) for row in flattened})
    horizons: dict[str, Any] = {}
    for horizon in ("H1", "H2", "H3"):
        horizons[horizon] = {
            "all": _diagnostic_group(flattened, horizon),
            "per_family": {
                family: _diagnostic_group(
                    [row for row in flattened if str(row["family"]) == family],
                    horizon,
                )
                for family in families
            },
            "per_role": {
                role: _diagnostic_group(
                    [row for row in flattened if str(row["role"]) == role],
                    horizon,
                )
                for role in roles
            },
        }
    monotonicity = {
        "all": _monotonic_group(flattened),
        "per_family": {
            family: _monotonic_group(
                [row for row in flattened if str(row["family"]) == family]
            )
            for family in families
        },
        "per_role": {
            role: _monotonic_group(
                [row for row in flattened if str(row["role"]) == role]
            )
            for role in roles
        },
    }
    per_candidate = []
    for row in flattened:
        costs = [float(row[key]) for key in ("cost_current", "cost_h1", "cost_h2", "cost_h3")]
        deltas = [costs[i] - costs[i + 1] for i in range(3)]
        nonincrease = [value >= -COST_TIE_TOLERANCE for value in deltas]
        strict = [value > COST_TIE_TOLERANCE for value in deltas]
        endpoint_delta = costs[0] - costs[3]
        per_candidate.append(
            {
                "state_id": row["state_id"],
                "candidate_index": _candidate_index(row),
                "family": row["family"],
                "role": row["role"],
                "current_to_h1_nonincrease": bool(nonincrease[0]),
                "h1_to_h2_nonincrease": bool(nonincrease[1]),
                "h2_to_h3_nonincrease": bool(nonincrease[2]),
                "current_to_h1_strict_decrease": bool(strict[0]),
                "h1_to_h2_strict_decrease": bool(strict[1]),
                "h2_to_h3_strict_decrease": bool(strict[2]),
                "current_to_h1_decrease": bool(strict[0]),
                "h1_to_h2_decrease": bool(strict[1]),
                "h2_to_h3_decrease": bool(strict[2]),
                "current_to_h3_nonincrease": bool(
                    endpoint_delta >= -COST_TIE_TOLERANCE
                ),
                "current_to_h3_strict_decrease": bool(
                    endpoint_delta > COST_TIE_TOLERANCE
                ),
                "current_to_h3_decrease": bool(
                    endpoint_delta > COST_TIE_TOLERANCE
                ),
                "overall_monotonic_trajectory": bool(all(nonincrease)),
                "overall_strict_decrease_trajectory": bool(all(strict)),
                "current_minus_h1": costs[0] - costs[1],
                "current_minus_h2": costs[0] - costs[2],
                "current_minus_h3": costs[0] - costs[3],
            }
        )
    return {
        "schema": "jepa_local_waypoint_latent_progress_diagnostics_v1",
        "source_id": source_id,
        "diagnostic_only": True,
        "gate": False,
        "horizons": horizons,
        "monotonicity": monotonicity,
        "per_candidate": per_candidate,
    }


def _binary_downranking_group(
    state_rows: Mapping[str, Sequence[tuple[Mapping[str, Any], float, int]]],
    label,
    *,
    adverse_name: str,
    reference_name: str,
) -> dict[str, Any]:
    ordered_pairs = 0
    credit = 0.0
    adverse_ranks: list[int] = []
    reference_ranks: list[int] = []
    for rows in state_rows.values():
        adverse = [(row, cost, rank) for row, cost, rank in rows if label(row)]
        reference = [(row, cost, rank) for row, cost, rank in rows if not label(row)]
        adverse_ranks.extend(rank for _, _, rank in adverse)
        reference_ranks.extend(rank for _, _, rank in reference)
        for _, adverse_cost, _ in adverse:
            for _, reference_cost, _ in reference:
                ordered_pairs += 1
                delta = adverse_cost - reference_cost
                if abs(delta) <= COST_TIE_TOLERANCE:
                    credit += 0.5
                elif delta > 0.0:
                    credit += 1.0

    def rank_summary(values: Sequence[int]) -> dict[str, Any]:
        return {
            "count": len(values),
            "mean_cost_rank": float(np.mean(values)) if values else None,
            "median_cost_rank": float(np.median(values)) if values else None,
        }

    return {
        "adverse_class": adverse_name,
        "reference_class": reference_name,
        "ordered_cross_class_pairs": ordered_pairs,
        "downranking_correct_credit": credit,
        "downranking_accuracy": credit / ordered_pairs if ordered_pairs else None,
        "cost_rank_by_class": {
            adverse_name: rank_summary(adverse_ranks),
            reference_name: rank_summary(reference_ranks),
        },
    }


def all_candidate_tendency_diagnostics(
    candidates_by_state: Mapping[str, Sequence[Mapping[str, Any]]],
    costs_by_state: Mapping[str, Any],
    *,
    source_id: str,
) -> dict[str, Any]:
    """Report non-classifying ALL_CANDIDATES contact/outcome cost tendencies."""

    if source_id not in SOURCE_IDS:
        raise PlanningCostMetricsError("tendency diagnostics require a latent source")
    if set(candidates_by_state) != set(costs_by_state):
        raise PlanningCostMetricsError("candidate/cost state identities must match")
    ranked: dict[str, list[tuple[Mapping[str, Any], float, int]]] = {}
    for state_id in candidates_by_state:
        rows = list(candidates_by_state[state_id])
        costs = np.asarray(costs_by_state[state_id], dtype=np.float64)
        if costs.shape != (len(rows),) or not np.isfinite(costs).all():
            raise PlanningCostMetricsError("tendency costs must align and be finite")
        order = _cost_order(costs, [_candidate_index(row) for row in rows])
        rank_by_position = {position: rank + 1 for rank, position in enumerate(order)}
        ranked[state_id] = [
            (row, float(costs[position]), rank_by_position[position])
            for position, row in enumerate(rows)
        ]

    def immediate_contact(row: Mapping[str, Any]) -> bool:
        if "immediate_contact_h1" in row:
            return _strict_bool(row["immediate_contact_h1"], name="immediate_contact_h1")
        return _oracle_contact(row)

    def group(subset: Mapping[str, Sequence[tuple[Mapping[str, Any], float, int]]]) -> dict[str, Any]:
        return {
            "immediate_contact_h1": _binary_downranking_group(
                subset,
                immediate_contact,
                adverse_name="contact",
                reference_name="contact_free",
            ),
            "successor_nonviable": _binary_downranking_group(
                subset,
                lambda row: not _successor_viable(row),
                adverse_name="nonviable",
                reference_name="viable",
            ),
            "stuck": _binary_downranking_group(
                subset,
                _stuck,
                adverse_name="stuck",
                reference_name="not_stuck",
            ),
            "no_progress": _binary_downranking_group(
                subset,
                lambda row: _distance_progress(row) <= 0.0,
                adverse_name="no_progress",
                reference_name="positive_progress",
            ),
            "stuck_or_no_progress": _binary_downranking_group(
                subset,
                lambda row: _stuck(row) or _distance_progress(row) <= 0.0,
                adverse_name="stuck_or_no_progress",
                reference_name="not_stuck_positive_progress",
            ),
        }

    families = sorted(
        {str(row[0].get("family", "")) for rows in ranked.values() for row in rows}
    )
    roles = sorted(
        {str(row[0].get("role", "")) for rows in ranked.values() for row in rows}
    )
    if "" in families or "" in roles:
        raise PlanningCostMetricsError("tendency rows must bind family and role")

    def select(key: str, value: str) -> dict[str, list[tuple[Mapping[str, Any], float, int]]]:
        output: dict[str, list[tuple[Mapping[str, Any], float, int]]] = {}
        for state_id, rows in ranked.items():
            kept = [row for row in rows if str(row[0][key]) == value]
            if kept:
                output[state_id] = kept
        return output

    return {
        "schema": "jepa_local_waypoint_all_candidate_tendency_diagnostics_v1",
        "source_id": source_id,
        "population_id": ALL_CANDIDATES,
        "diagnostic_only": True,
        "gate": False,
        "all": group(ranked),
        "per_family": {family: group(select("family", family)) for family in families},
        "per_role": {role: group(select("role", role)) for role in roles},
    }


def _state_map(source: Mapping[str, Any], population_id: str) -> dict[str, Mapping[str, Any]]:
    try:
        rows = source["populations"][population_id]["per_state"]
    except (KeyError, TypeError) as exc:
        raise PlanningCostMetricsError(
            f"source lacks {population_id} per-state metrics"
        ) from exc
    output = {str(row["state_id"]): row for row in rows}
    if len(output) != len(rows):
        raise PlanningCostMetricsError("source has duplicate per-state identities")
    return output


def _validate_bootstrap_parameters(*, draws: int, seed: int) -> None:
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise PlanningCostMetricsError("bootstrap draws must be a positive integer")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**64
    ):
        raise PlanningCostMetricsError("bootstrap seed must be an unsigned uint64")


def _hash_bootstrap_indices(
    *,
    state_count: int,
    sample_size: int,
    draws: int,
    seed: int,
    comparison_id: str,
) -> tuple[np.ndarray, str]:
    """Exact STATE_BOOTSTRAP_V1 hash resampling plan."""

    _validate_bootstrap_parameters(draws=draws, seed=seed)
    if state_count <= 0 or sample_size <= 0:
        raise PlanningCostMetricsError("bootstrap state and sample counts must be positive")
    if not isinstance(comparison_id, str) or not comparison_id:
        raise PlanningCostMetricsError("comparison_id must be a nonempty string")
    prefix = (
        STATE_BOOTSTRAP_NAMESPACE.encode("utf-8")
        + b"\x00"
        + seed.to_bytes(8, "big", signed=False)
        + b"\x00"
        + comparison_id.encode("utf-8")
        + b"\x00"
    )
    indices = np.empty((draws, sample_size), dtype=np.int64)
    plan_digest = hashlib.sha256()
    for replicate in range(draws):
        replicate_bytes = replicate.to_bytes(4, "big", signed=False)
        for draw in range(sample_size):
            digest = hashlib.sha256(
                prefix
                + replicate_bytes
                + draw.to_bytes(4, "big", signed=False)
            ).digest()
            indices[replicate, draw] = int.from_bytes(
                digest[:8], "big", signed=False
            ) % state_count
    plan_digest.update(prefix)
    plan_digest.update(np.asarray(indices, dtype=">i8").tobytes(order="C"))
    return indices, plan_digest.hexdigest()


def type7_quantile(values: Any, probability: float) -> float:
    """Exact sorted linear Type-7 quantile used by the frozen bootstrap."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise PlanningCostMetricsError("quantile values must be a finite nonempty vector")
    p = _finite_scalar(probability, name="probability")
    if not 0.0 <= p <= 1.0:
        raise PlanningCostMetricsError("quantile probability must be in [0,1]")
    ordered = np.sort(array, kind="stable")
    h = (len(ordered) - 1) * p
    lower = math.floor(h)
    upper = math.ceil(h)
    fraction = h - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def _bootstrap_summary(
    point: float,
    replicates: np.ndarray,
    *,
    family_points: Mapping[str, float],
    plan_digest: str,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    lower = type7_quantile(replicates, BOOTSTRAP_PERCENTILES[0] / 100.0)
    upper = type7_quantile(replicates, BOOTSTRAP_PERCENTILES[1] / 100.0)
    return {
        "point": float(point),
        "bootstrap_lower_95": float(lower),
        "bootstrap_upper_95": float(upper),
        "per_family_point": dict(sorted(family_points.items())),
        "draws": draws,
        "seed": seed,
        "percentiles": list(BOOTSTRAP_PERCENTILES),
        "percentile_method": "Type-7 linear",
        "unit": "paired state resampling",
        "shared_plan_sha256": plan_digest,
    }


def paired_family_state_bootstrap(
    candidate_by_state: Mapping[str, float],
    comparator_by_state: Mapping[str, float],
    family_by_state: Mapping[str, str],
    *,
    comparison_id: str = "PAIRED_EQUAL_FAMILY_DIFFERENCE",
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap an equal-family candidate-minus-comparator state mean.

    This specialised reducer is retained for prospectively equal-family
    quantities.  Use :func:`paired_state_bootstrap` for the ordinary pooled
    state mean used by progress and normalised-regret materiality.
    """

    identities = set(candidate_by_state)
    if identities != set(comparator_by_state) or identities != set(family_by_state):
        raise PlanningCostMetricsError("paired bootstrap state identities must match")
    if not identities:
        raise PlanningCostMetricsError("paired bootstrap requires at least one state")
    state_order = [state for state in candidate_by_state if state in identities]
    family_points: dict[str, float] = {}
    family_replicates: list[np.ndarray] = []
    plan_digests: dict[str, str] = {}
    for family in sorted({str(family_by_state[state]) for state in state_order}):
        states = [state for state in state_order if str(family_by_state[state]) == family]
        differences = np.asarray(
            [
                _finite_scalar(candidate_by_state[state], name="candidate")
                - _finite_scalar(comparator_by_state[state], name="comparator")
                for state in states
            ],
            dtype=np.float64,
        )
        indices, digest = _hash_bootstrap_indices(
            state_count=len(states),
            sample_size=len(states),
            draws=draws,
            seed=seed,
            comparison_id=f"{comparison_id}/{family}",
        )
        family_points[family] = float(np.mean(differences))
        family_replicates.append(np.mean(differences[indices], axis=1))
        plan_digests[family] = digest
    replicates = np.mean(np.stack(family_replicates, axis=1), axis=1)
    combined_digest = canonical_sha256(plan_digests)
    result = _bootstrap_summary(
        float(np.mean(list(family_points.values()))),
        replicates,
        family_points=family_points,
        plan_digest=combined_digest,
        draws=draws,
        seed=seed,
    )
    result["unit"] = "paired states within family; equal-family mean"
    result["comparison_id"] = comparison_id
    result["per_family_plan_sha256"] = plan_digests
    return result


def paired_state_bootstrap(
    candidate_by_state: Mapping[str, float],
    comparator_by_state: Mapping[str, float],
    family_by_state: Mapping[str, str],
    *,
    comparison_id: str = "PAIRED_STATE_DIFFERENCE",
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap the pooled paired-state mean with STATE_BOOTSTRAP_V1."""

    identities = set(candidate_by_state)
    if identities != set(comparator_by_state) or identities != set(family_by_state):
        raise PlanningCostMetricsError("paired bootstrap state identities must match")
    if not identities:
        raise PlanningCostMetricsError("paired bootstrap requires at least one state")
    _validate_bootstrap_parameters(draws=draws, seed=seed)
    # Mapping insertion order is the frozen state-manifest order supplied by
    # the evaluator; sorting here would silently change the registered draw.
    states = [state for state in candidate_by_state if state in identities]
    differences = np.asarray(
        [
            _finite_scalar(candidate_by_state[state], name="candidate")
            - _finite_scalar(comparator_by_state[state], name="comparator")
            for state in states
        ],
        dtype=np.float64,
    )
    indices, plan_digest = _hash_bootstrap_indices(
        state_count=len(states),
        sample_size=len(states),
        draws=draws,
        seed=seed,
        comparison_id=comparison_id,
    )
    per_family = {
        family: float(
            np.mean(
                [
                    differences[position]
                    for position, state in enumerate(states)
                    if str(family_by_state[state]) == family
                ]
            )
        )
        for family in sorted({str(family_by_state[state]) for state in states})
    }
    summary = _bootstrap_summary(
        float(np.mean(differences)),
        np.mean(differences[indices], axis=1),
        family_points=per_family,
        plan_digest=plan_digest,
        draws=draws,
        seed=seed,
    )
    summary["unit"] = "paired state resampling; pooled state mean"
    summary["states"] = len(states)
    summary["state_order"] = list(states)
    summary["comparison_id"] = comparison_id
    return summary


def _unavailable_bootstrap_summary(
    *, draws: int, seed: int, reason: str
) -> dict[str, Any]:
    return {
        "point": None,
        "bootstrap_lower_95": None,
        "bootstrap_upper_95": None,
        "per_family_point": {},
        "draws": draws,
        "seed": seed,
        "percentiles": list(BOOTSTRAP_PERCENTILES),
        "percentile_method": "Type-7 linear",
        "unit": "paired state resampling; pooled state mean",
        "shared_plan_sha256": None,
        "states": 0,
        "unavailable_reason": reason,
    }


def _family_ratio_replicates(
    candidate: Mapping[str, Mapping[str, Any]],
    comparator: Mapping[str, Mapping[str, Any]],
    *,
    included_families: Sequence[str],
    draws: int,
    seed: int,
    comparison_id: str,
) -> tuple[float, np.ndarray, dict[str, float], dict[str, str]]:
    family_points: dict[str, float] = {}
    sampled: list[np.ndarray] = []
    plan_digests: dict[str, str] = {}
    for family in included_families:
        states = [
            state for state in candidate if str(candidate[state]["family"]) == family
        ]
        if not states:
            raise PlanningCostMetricsError(f"hard family {family} has no paired states")
        indices, plan_digest = _hash_bootstrap_indices(
            state_count=len(states),
            sample_size=len(states),
            draws=draws,
            seed=seed,
            comparison_id=f"{comparison_id}/{family}",
        )
        plan_digests[family] = plan_digest
        candidate_selected = np.asarray(
            [candidate[state]["selected_route_progress_m"] for state in states], np.float64
        )
        comparator_selected = np.asarray(
            [comparator[state]["selected_route_progress_m"] for state in states], np.float64
        )
        oracle_best = np.asarray(
            [candidate[state]["oracle_best_route_progress_m"] for state in states], np.float64
        )
        if not np.allclose(
            oracle_best,
            np.asarray(
                [comparator[state]["oracle_best_route_progress_m"] for state in states],
                np.float64,
            ),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise PlanningCostMetricsError("paired sources disagree on oracle-best progress")
        denominator = max(abs(float(np.sum(oracle_best))), NUMERIC_EPSILON)
        family_points[family] = float(
            (np.sum(candidate_selected) - np.sum(comparator_selected)) / denominator
        )
        sampled_denominator = np.maximum(
            np.abs(np.sum(oracle_best[indices], axis=1)), NUMERIC_EPSILON
        )
        sampled.append(
            (
                np.sum(candidate_selected[indices], axis=1)
                - np.sum(comparator_selected[indices], axis=1)
            )
            / sampled_denominator
        )
    return (
        float(np.mean(list(family_points.values()))),
        np.mean(np.stack(sampled, axis=1), axis=1),
        family_points,
        plan_digests,
    )


def paired_source_comparison(
    candidate_source: Mapping[str, Any],
    comparator_source: Mapping[str, Any],
    *,
    population_id: str = ORACLE_VIABILITY_ADMISSIBLE,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute frozen paired improvements and their descriptive bootstrap CIs."""

    candidate = _state_map(candidate_source, population_id)
    comparator = _state_map(comparator_source, population_id)
    if set(candidate) != set(comparator):
        raise PlanningCostMetricsError("paired sources have different state identities")
    family_by_state = {state: str(candidate[state]["family"]) for state in candidate}
    if any(str(comparator[state]["family"]) != family_by_state[state] for state in candidate):
        raise PlanningCostMetricsError("paired source family identities differ")
    candidate_id = str(candidate_source.get("source_id"))
    comparator_id = str(comparator_source.get("source_id"))
    comparison_prefix = f"{candidate_id}_MINUS_{comparator_id}/{population_id}"
    progress = paired_state_bootstrap(
        {
            state: float(candidate[state]["selected_route_progress_m"])
            for state in candidate
        },
        {
            state: float(comparator[state]["selected_route_progress_m"])
            for state in comparator
        },
        family_by_state,
        comparison_id=f"{comparison_prefix}/SELECTED_PROGRESS_M",
        draws=draws,
        seed=seed,
    )

    # Regret reduction is comparator minus candidate, so positive is favorable.
    regret_states = [
        state
        for state in candidate
        if candidate[state]["normalized_regret"] is not None
        and comparator[state]["normalized_regret"] is not None
    ]
    if regret_states:
        regret = paired_state_bootstrap(
            {
                state: float(comparator[state]["normalized_regret"])
                for state in regret_states
            },
            {
                state: float(candidate[state]["normalized_regret"])
                for state in regret_states
            },
            {state: family_by_state[state] for state in regret_states},
            comparison_id=(
                f"{comparator_id}_MINUS_{candidate_id}/{population_id}/NORMALIZED_REGRET"
            ),
            draws=draws,
            seed=seed,
        )
    else:
        regret = _unavailable_bootstrap_summary(
            draws=draws,
            seed=seed,
            reason="no jointly nonabstaining state has defined normalized regret",
        )

    hard_point, hard_replicates, hard_families, hard_plan_digests = _family_ratio_replicates(
        candidate,
        comparator,
        included_families=HARD_FAMILY_IDS,
        draws=draws,
        seed=seed,
        comparison_id=f"{comparison_prefix}/HARD_FAMILY_PROGRESS_RATIO",
    )
    hard_plan_digest = canonical_sha256(hard_plan_digests)
    hard = _bootstrap_summary(
        hard_point,
        hard_replicates,
        family_points=hard_families,
        plan_digest=hard_plan_digest,
        draws=draws,
        seed=seed,
    )
    hard["per_family_plan_sha256"] = hard_plan_digests

    candidate_all = candidate_source["populations"][ALL_CANDIDATES]["aggregate"]
    comparator_all = comparator_source["populations"][ALL_CANDIDATES]["aggregate"]
    no_contact_increase = _aggregate_count(
        candidate_all, "selected_immediate_contacts_h1", "selected_oracle_contacts"
    ) <= _aggregate_count(
        comparator_all, "selected_immediate_contacts_h1", "selected_oracle_contacts"
    )
    no_nonviable_increase = _aggregate_count(
        candidate_all, "selected_nonviable_successors", "selected_oracle_nonviable"
    ) <= _aggregate_count(
        comparator_all, "selected_nonviable_successors", "selected_oracle_nonviable"
    )
    candidate_hard_collapsed = any(
        candidate_source["populations"][population_id]["per_family"][family][
            "complete_family_collapse"
        ]
        for family in HARD_FAMILY_IDS
    )
    comparator_hard_collapsed = any(
        comparator_source["populations"][population_id]["per_family"][family][
            "complete_family_collapse"
        ]
        for family in HARD_FAMILY_IDS
    )
    progress_trigger = bool(
        progress["point"] >= MATERIAL_IMPROVEMENT_THRESHOLDS["mean_route_progress_gain_m"]
        and progress["bootstrap_lower_95"] > 0.0
    )
    regret_trigger = bool(
        regret["point"] is not None
        and regret["point"]
        >= MATERIAL_IMPROVEMENT_THRESHOLDS["normalized_regret_reduction"]
        and regret["bootstrap_lower_95"] is not None
        and regret["bootstrap_lower_95"] > 0.0
    )
    hard_trigger = bool(
        hard["point"]
        >= MATERIAL_IMPROVEMENT_THRESHOLDS["hard_family_progress_ratio_gain"]
        and hard["bootstrap_lower_95"] > 0.0
    )
    safety_preserved = bool(
        no_contact_increase
        and no_nonviable_increase
        and not candidate_hard_collapsed
        and not comparator_hard_collapsed
    )
    progress_material = progress_trigger and safety_preserved
    regret_material = regret_trigger and safety_preserved
    hard_material = hard_trigger and safety_preserved

    def optional_scalar(row: Mapping[str, Any], *keys: str) -> float | None:
        for key in keys:
            if key in row and row[key] is not None:
                return _finite_scalar(row[key], name=key)
        return None

    def optional_candidate_index(row: Mapping[str, Any]) -> int | None:
        value = row.get("selected_candidate_index")
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise PlanningCostMetricsError("selected_candidate_index must be an integer")
        return int(value)

    paired_state_rows = []
    for state_id in candidate:
        candidate_row = candidate[state_id]
        comparator_row = comparator[state_id]
        candidate_regret = optional_scalar(
            candidate_row, "normalized_regret", "normalised_regret"
        )
        comparator_regret = optional_scalar(
            comparator_row, "normalized_regret", "normalised_regret"
        )
        candidate_pairwise = optional_scalar(candidate_row, "pairwise_accuracy")
        comparator_pairwise = optional_scalar(comparator_row, "pairwise_accuracy")
        candidate_top3 = candidate_row.get("best_route_top3")
        comparator_top3 = comparator_row.get("best_route_top3")
        if candidate_top3 is not None:
            candidate_top3 = _strict_bool(
                candidate_top3, name="candidate.best_route_top3"
            )
        if comparator_top3 is not None:
            comparator_top3 = _strict_bool(
                comparator_top3, name="comparator.best_route_top3"
            )
        candidate_rank = optional_scalar(
            candidate_row, "best_route_rank", "oracle_best_cost_rank"
        )
        comparator_rank = optional_scalar(
            comparator_row, "best_route_rank", "oracle_best_cost_rank"
        )
        candidate_progress = _finite_scalar(
            candidate_row["selected_route_progress_m"],
            name="candidate.selected_route_progress_m",
        )
        comparator_progress = _finite_scalar(
            comparator_row["selected_route_progress_m"],
            name="comparator.selected_route_progress_m",
        )
        paired_state_rows.append(
            {
                "state_id": state_id,
                "family": family_by_state[state_id],
                "candidate_selected_candidate_index": optional_candidate_index(
                    candidate_row
                ),
                "comparator_selected_candidate_index": optional_candidate_index(
                    comparator_row
                ),
                "candidate_selected_progress_m": candidate_progress,
                "comparator_selected_progress_m": comparator_progress,
                "selected_progress_delta_m": candidate_progress
                - comparator_progress,
                "candidate_normalized_regret": candidate_regret,
                "comparator_normalized_regret": comparator_regret,
                "normalized_regret_reduction": (
                    None
                    if candidate_regret is None or comparator_regret is None
                    else comparator_regret - candidate_regret
                ),
                "candidate_pairwise_accuracy": candidate_pairwise,
                "comparator_pairwise_accuracy": comparator_pairwise,
                "pairwise_accuracy_delta": (
                    None
                    if candidate_pairwise is None or comparator_pairwise is None
                    else candidate_pairwise - comparator_pairwise
                ),
                "candidate_best_route_top3": candidate_top3,
                "comparator_best_route_top3": comparator_top3,
                "best_route_top3_delta": (
                    None
                    if candidate_top3 is None or comparator_top3 is None
                    else int(candidate_top3) - int(comparator_top3)
                ),
                "candidate_best_route_rank": candidate_rank,
                "comparator_best_route_rank": comparator_rank,
                "best_route_rank_improvement": (
                    None
                    if candidate_rank is None or comparator_rank is None
                    else comparator_rank - candidate_rank
                ),
            }
        )
    return {
        "schema": "jepa_local_waypoint_paired_source_comparison_v1",
        "candidate_source_id": candidate_source.get("source_id"),
        "comparator_source_id": comparator_source.get("source_id"),
        "population_id": population_id,
        "mean_route_progress_gain_m": progress,
        "normalized_regret_reduction": regret,
        "hard_family_progress_ratio_gain": hard,
        "per_state": paired_state_rows,
        "no_all_candidate_contact_selection_increase": no_contact_increase,
        "no_all_candidate_nonviable_selection_increase": no_nonviable_increase,
        "candidate_hard_family_collapse": candidate_hard_collapsed,
        "comparator_hard_family_collapse": comparator_hard_collapsed,
        "required_safety_and_support_preserved": safety_preserved,
        "raw_material_triggers": {
            "mean_route_progress": progress_trigger,
            "normalized_regret": regret_trigger,
            "hard_family_progress_ratio": hard_trigger,
        },
        "material_criteria": {
            "mean_route_progress": progress_material,
            "normalized_regret": regret_material,
            "hard_family_progress_ratio": hard_material,
        },
        "material_improvement": progress_material or regret_material or hard_material,
        "bootstrap": {
            "draws": draws,
            "seed": seed,
            "percentiles": list(BOOTSTRAP_PERCENTILES),
            "descriptive_only": True,
            "candidate_rows_bootstrapped_independently": False,
            "hard_family_shared_plan_sha256": hard_plan_digest,
            "progress_plan_sha256": progress["shared_plan_sha256"],
            "regret_plan_sha256": regret["shared_plan_sha256"],
        },
    }


def _criterion(value: Any, operator: str, threshold: Any, passed: bool) -> dict[str, Any]:
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _aggregate_count(aggregate: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        if key in aggregate:
            value = aggregate[key]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise PlanningCostMetricsError(f"{key} must be an integer count")
            if int(value) < 0:
                raise PlanningCostMetricsError(f"{key} cannot be negative")
            return int(value)
    raise PlanningCostMetricsError(f"aggregate lacks required count aliases {keys}")


def _population_summary(source: Mapping[str, Any], population_id: str) -> Mapping[str, Any]:
    try:
        return source["populations"][population_id]
    except (KeyError, TypeError) as exc:
        raise PlanningCostMetricsError(f"source lacks population {population_id}") from exc


def _verified_no_family_complete_collapse(population: Mapping[str, Any]) -> bool:
    try:
        per_family = population["per_family"]
    except (KeyError, TypeError) as exc:
        raise PlanningCostMetricsError("gate population lacks per-family metrics") from exc
    if set(per_family) != set(FAMILY_IDS):
        raise PlanningCostMetricsError(
            "gate population must contain exactly the four frozen families"
        )
    derived = all(
        not _strict_bool(
            per_family[family]["complete_family_collapse"],
            name=f"{family}.complete_family_collapse",
        )
        for family in FAMILY_IDS
    )
    reported = _strict_bool(
        population.get("no_family_complete_collapse"),
        name="no_family_complete_collapse",
    )
    if reported != derived:
        raise PlanningCostMetricsError("reported family-collapse summary is inconsistent")
    return derived


def evaluate_true_future_gate(source: Mapping[str, Any]) -> dict[str, Any]:
    """Apply every immutable true-future gate under oracle viability."""

    if source.get("source_id") not in {TRUE_FUTURE, TRUE_FUTURE_LATENT_COST}:
        raise PlanningCostMetricsError("true-future gate requires the true-future source")
    population = _population_summary(source, ORACLE_VIABILITY_ADMISSIBLE)
    aggregate = population["aggregate"]
    pairwise = aggregate["pairwise_accuracy"]
    spearman = aggregate["spearman_rho"]
    regret = aggregate["normalized_regret"]
    top3 = aggregate["best_route_top3_rate"]
    progress_ratio = aggregate["selected_progress_ratio"]
    no_collapse = _verified_no_family_complete_collapse(population)
    criteria = {
        "pairwise_accuracy": _criterion(
            pairwise,
            ">=",
            0.70,
            pairwise is not None and float(pairwise) >= 0.70,
        ),
        "spearman_rho": _criterion(
            spearman,
            ">=",
            0.60,
            spearman is not None and float(spearman) >= 0.60,
        ),
        "normalized_regret": _criterion(
            regret,
            "<=",
            0.25,
            regret is not None and float(regret) <= 0.25,
        ),
        "best_route_top3": _criterion(
            top3,
            ">=",
            0.75,
            top3 is not None and float(top3) >= 0.75,
        ),
        "selected_progress_ratio": _criterion(
            progress_ratio, ">=", 0.80, float(progress_ratio) >= 0.80
        ),
        "no_family_complete_collapse": _criterion(
            no_collapse, "==", True, no_collapse
        ),
    }
    passed = all(row["passed"] for row in criteria.values())
    return {
        "schema": "jepa_local_waypoint_true_future_gate_v1",
        "population_id": ORACLE_VIABILITY_ADMISSIBLE,
        "criteria": criteria,
        "pass": passed,
        "classification": (
            "TRUE_FUTURE_LATENT_GOAL_COST_SIGNAL"
            if passed
            else "TRUE_FUTURE_LATENT_GOAL_COST_NO_GO"
        ),
    }


def evaluate_absolute_base_preservation_screen(
    source: Mapping[str, Any],
    true_future_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Report the non-classifying absolute screen for either predicted source."""

    source_id = source.get("source_id")
    if source_id not in {
        ONE_STEP_PREDICTED,
        ONE_STEP_PREDICTED_LATENT_COST,
        TWO_STEP_PREDICTED,
        TWO_STEP_PREDICTED_LATENT_COST,
    }:
        raise PlanningCostMetricsError(
            "absolute base-preservation screen requires a predicted source"
        )
    if true_future_source.get("source_id") not in {
        TRUE_FUTURE,
        TRUE_FUTURE_LATENT_COST,
    }:
        raise PlanningCostMetricsError("true-future source identity is invalid")
    population = _population_summary(source, ORACLE_VIABILITY_ADMISSIBLE)
    aggregate = population["aggregate"]
    true = _population_summary(
        true_future_source, ORACLE_VIABILITY_ADMISSIBLE
    )["aggregate"]
    progress_ratio = float(aggregate["selected_route_progress_m_sum"]) / max(
        abs(float(true["selected_route_progress_m_sum"])), NUMERIC_EPSILON
    )
    pairwise = aggregate["pairwise_accuracy"]
    regret = aggregate["normalized_regret"]
    criteria = {
        "pairwise_accuracy": _criterion(
            pairwise,
            ">=",
            0.65,
            pairwise is not None and float(pairwise) >= 0.65,
        ),
        "normalized_regret": _criterion(
            regret,
            "<=",
            0.30,
            regret is not None and float(regret) <= 0.30,
        ),
        "selected_progress_fraction_of_true_future": _criterion(
            progress_ratio, ">=", 0.75, progress_ratio >= 0.75
        ),
        "no_family_complete_collapse": _criterion(
            _verified_no_family_complete_collapse(population),
            "==",
            True,
            _verified_no_family_complete_collapse(population),
        ),
    }
    return {
        "schema": "jepa_local_waypoint_absolute_base_preservation_screen_v1",
        "source_id": source_id,
        "population_id": ORACLE_VIABILITY_ADMISSIBLE,
        "diagnostic_only": True,
        "criteria": criteria,
        "pass": all(row["passed"] for row in criteria.values()),
    }


def evaluate_predicted_gate(
    *,
    true_future_source: Mapping[str, Any],
    one_step_source: Mapping[str, Any],
    two_step_source: Mapping[str, Any],
    true_future_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the immutable two-step predicted-planning gate."""

    if true_future_source.get("source_id") not in {
        TRUE_FUTURE,
        TRUE_FUTURE_LATENT_COST,
    }:
        raise PlanningCostMetricsError("true-future source identity is invalid")
    if one_step_source.get("source_id") not in {
        ONE_STEP_PREDICTED,
        ONE_STEP_PREDICTED_LATENT_COST,
    } or two_step_source.get("source_id") not in {
        TWO_STEP_PREDICTED,
        TWO_STEP_PREDICTED_LATENT_COST,
    }:
        raise PlanningCostMetricsError("predicted gate source identities are invalid")
    true_gate = true_future_gate or evaluate_true_future_gate(true_future_source)
    one = _population_summary(one_step_source, ORACLE_VIABILITY_ADMISSIBLE)["aggregate"]
    two_population = _population_summary(two_step_source, ORACLE_VIABILITY_ADMISSIBLE)
    two = two_population["aggregate"]
    true = _population_summary(true_future_source, ORACLE_VIABILITY_ADMISSIBLE)[
        "aggregate"
    ]
    one_all = _population_summary(one_step_source, ALL_CANDIDATES)["aggregate"]
    two_all = _population_summary(two_step_source, ALL_CANDIDATES)["aggregate"]
    two_to_true_progress = float(two["selected_route_progress_m_sum"]) / max(
        abs(float(true["selected_route_progress_m_sum"])), NUMERIC_EPSILON
    )
    pairwise_improvement = (
        two["pairwise_accuracy"] is not None
        and one["pairwise_accuracy"] is not None
        and float(two["pairwise_accuracy"]) > float(one["pairwise_accuracy"])
    )
    regret_improvement = (
        two["normalized_regret"] is not None
        and one["normalized_regret"] is not None
        and float(two["normalized_regret"]) < float(one["normalized_regret"])
    )
    progress_improvement = float(two["selected_route_progress_m_sum"]) > float(
        one["selected_route_progress_m_sum"]
    )
    two_contacts = _aggregate_count(
        two_all, "selected_immediate_contacts_h1", "selected_oracle_contacts"
    )
    one_contacts = _aggregate_count(
        one_all, "selected_immediate_contacts_h1", "selected_oracle_contacts"
    )
    two_nonviable = _aggregate_count(
        two_all, "selected_nonviable_successors", "selected_oracle_nonviable"
    )
    one_nonviable = _aggregate_count(
        one_all, "selected_nonviable_successors", "selected_oracle_nonviable"
    )
    no_contact_increase = two_contacts <= one_contacts
    no_nonviable_increase = two_nonviable <= one_nonviable
    criteria = {
        "true_future_gate_passes": _criterion(
            bool(true_gate.get("pass")), "==", True, bool(true_gate.get("pass"))
        ),
        "two_step_pairwise_accuracy": _criterion(
            two["pairwise_accuracy"],
            ">=",
            0.65,
            two["pairwise_accuracy"] is not None
            and float(two["pairwise_accuracy"]) >= 0.65,
        ),
        "two_step_normalized_regret": _criterion(
            two["normalized_regret"],
            "<=",
            0.30,
            two["normalized_regret"] is not None
            and float(two["normalized_regret"]) <= 0.30,
        ),
        "two_step_to_true_selected_progress_ratio": _criterion(
            two_to_true_progress, ">=", 0.75, two_to_true_progress >= 0.75
        ),
        "all_candidates_contact_selections_not_increased": _criterion(
            [two_contacts, one_contacts],
            "two_step <= one_step",
            True,
            no_contact_increase,
        ),
        "all_candidates_nonviable_selections_not_increased": _criterion(
            [two_nonviable, one_nonviable],
            "two_step <= one_step",
            True,
            no_nonviable_increase,
        ),
        "pairwise_improves_over_one_step": _criterion(
            (
                None
                if two["pairwise_accuracy"] is None or one["pairwise_accuracy"] is None
                else float(two["pairwise_accuracy"]) - float(one["pairwise_accuracy"])
            ),
            ">",
            0.0,
            pairwise_improvement,
        ),
        "regret_or_progress_improves_over_one_step": _criterion(
            {"regret": regret_improvement, "progress": progress_improvement},
            "any",
            True,
            regret_improvement or progress_improvement,
        ),
        "no_family_complete_collapse": _criterion(
            _verified_no_family_complete_collapse(two_population),
            "==",
            True,
            _verified_no_family_complete_collapse(two_population),
        ),
    }
    passed = all(row["passed"] for row in criteria.values())
    one_step_screen = evaluate_absolute_base_preservation_screen(
        one_step_source, true_future_source
    )
    two_step_screen = evaluate_absolute_base_preservation_screen(
        two_step_source, true_future_source
    )
    return {
        "schema": "jepa_local_waypoint_predicted_gate_v1",
        "population_id": ORACLE_VIABILITY_ADMISSIBLE,
        "criteria": criteria,
        "pass": passed,
        "nonclassifying_absolute_screens": {
            "ONE_STEP_PREDICTED": one_step_screen,
            "TWO_STEP_PREDICTED": two_step_screen,
        },
    }


def next_experiment(primary_classification: str) -> str:
    """Map a terminal primary classification to exactly one registered successor."""

    mapping = {
        "TWO_STEP_JEPA_PLANNING_COST_SIGNAL": "ORACLE_ADMISSIBLE_CLOSED_LOOP_JEPA_MPC_V1",
        "TRUE_FUTURE_COST_SIGNAL_PREDICTOR_PLANNING_NO_GO": "PLAN_AWARE_MONOTONE_JEPA_COST_V1",
        "RAW_LATENT_GOAL_COST_NO_GO": "PLAN_AWARE_MONOTONE_JEPA_COST_V1",
        "KINEMATIC_BASELINE_DOMINANT": "NON_GREEDY_LOCAL_SUBGOAL_JEPA_PLANNING_V1",
    }
    if primary_classification not in mapping:
        raise PlanningCostMetricsError(
            f"unknown primary classification {primary_classification!r}"
        )
    return mapping[primary_classification]


def classify_qualification(
    *,
    true_future_gate: Mapping[str, Any],
    predicted_gate: Mapping[str, Any],
    jepa_vs_kinematic: Mapping[str, Any],
    kinematic_vs_jepa: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen primary precedence and incremental-value secondary."""

    true_pass = _strict_bool(true_future_gate.get("pass"), name="true_future_gate.pass")
    predicted_pass = _strict_bool(predicted_gate.get("pass"), name="predicted_gate.pass")
    jepa_incremental = _strict_bool(
        jepa_vs_kinematic.get("material_improvement"),
        name="jepa_vs_kinematic.material_improvement",
    )
    kinematic_superior = _strict_bool(
        kinematic_vs_jepa.get("material_improvement"),
        name="kinematic_vs_jepa.material_improvement",
    )
    expected_true_classification = (
        "TRUE_FUTURE_LATENT_GOAL_COST_SIGNAL"
        if true_pass
        else "TRUE_FUTURE_LATENT_GOAL_COST_NO_GO"
    )
    reported_true_classification = true_future_gate.get("classification")
    if reported_true_classification is not None and (
        reported_true_classification != expected_true_classification
    ):
        raise PlanningCostMetricsError(
            "true-future gate classification is inconsistent with its pass flag"
        )
    try:
        predicted_base_screens = predicted_gate[
            "nonclassifying_absolute_screens"
        ]
        one_step_screen_passed = _strict_bool(
            predicted_base_screens["ONE_STEP_PREDICTED"].get("pass"),
            name="ONE_STEP_PREDICTED base screen pass",
        )
        two_step_screen_passed = _strict_bool(
            predicted_base_screens["TWO_STEP_PREDICTED"].get("pass"),
            name="TWO_STEP_PREDICTED base screen pass",
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise PlanningCostMetricsError(
            "predicted gate lacks both frozen absolute base-preservation screens"
        ) from exc
    if not true_pass:
        primary = "RAW_LATENT_GOAL_COST_NO_GO"
    elif predicted_pass and kinematic_superior and not jepa_incremental:
        primary = "KINEMATIC_BASELINE_DOMINANT"
    elif predicted_pass:
        primary = "TWO_STEP_JEPA_PLANNING_COST_SIGNAL"
    else:
        primary = "TRUE_FUTURE_COST_SIGNAL_PREDICTOR_PLANNING_NO_GO"
    secondary = [SECONDARY_INCREMENTAL_VALUE] if jepa_incremental else []
    return {
        "schema": "jepa_local_waypoint_planning_cost_classification_v1",
        "primary_classification": primary,
        "secondary_classifications": secondary,
        "true_future_gate_classification": expected_true_classification,
        "two_step_gate_passed": predicted_pass,
        "two_step_gate_signal_or_null": (
            "TWO_STEP_JEPA_PLANNING_COST_SIGNAL" if predicted_pass else None
        ),
        "predicted_base_screens": {
            "ONE_STEP_PREDICTED": dict(
                predicted_base_screens["ONE_STEP_PREDICTED"]
            ),
            "TWO_STEP_PREDICTED": dict(
                predicted_base_screens["TWO_STEP_PREDICTED"]
            ),
        },
        "diagnostic_flags": {
            "both_predicted_base_screens_failed": (
                not one_step_screen_passed and not two_step_screen_passed
            ),
            "ONE_STEP_BASE_SCREEN_ONLY": (
                one_step_screen_passed and not two_step_screen_passed
            ),
            "two_step_base_screen_passed_but_full_gate_failed": (
                two_step_screen_passed and not predicted_pass
            ),
        },
        "precedence": [
            "true_future_gate_failure",
            "predicted_pass_and_kinematic_material_superiority_without_jepa_increment",
            "predicted_gate_pass",
            "true_future_signal_but_predicted_gate_failure",
        ],
        "next_experiment": next_experiment(primary),
    }


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic finite JSON bytes for synthetic/reporting receipts."""

    def ready(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): ready(item[key]) for key in sorted(item, key=str)}
        if isinstance(item, np.ndarray):
            return ready(item.tolist())
        if isinstance(item, np.generic):
            return ready(item.item())
        if isinstance(item, (list, tuple)):
            return [ready(child) for child in item]
        if isinstance(item, float):
            if not math.isfinite(item):
                raise PlanningCostMetricsError("canonical JSON forbids NaN/Infinity")
            return item
        if item is None or isinstance(item, (str, int, bool)):
            return item
        raise PlanningCostMetricsError(
            f"unsupported canonical JSON type {type(item).__name__}"
        )

    return (
        json.dumps(
            ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


__all__ = [
    "ALL_CANDIDATES",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "COMPARATOR_IDS",
    "COST_TIE_TOLERANCE",
    "DISTANCE_PREFERENCE_MARGIN_M",
    "EXPERIMENT_ID",
    "FAMILY_IDS",
    "HARD_FAMILY_IDS",
    "HEADING_PREFERENCE_MARGIN_RAD",
    "KINEMATIC_ROUTE_BASELINE",
    "L2_NORMALIZE_FLOOR",
    "LAYER_NORM_EPSILON",
    "NEXT_EXPERIMENTS",
    "ONE_STEP_PREDICTED",
    "ONE_STEP_PREDICTED_LATENT_COST",
    "ORACLE_CONTACT_FREE",
    "ORACLE_VIABILITY_ADMISSIBLE",
    "POPULATION_IDS",
    "PRIMARY_CLASSIFICATIONS",
    "PlanningCostMetricsError",
    "RANDOM",
    "RANDOM_NAMESPACE",
    "SECONDARY_INCREMENTAL_VALUE",
    "SOURCE_IDS",
    "STATE_BOOTSTRAP_NAMESPACE",
    "TOKEN_COST_DEFINITION",
    "TOKEN_COST_RANGE",
    "TRUE_FUTURE",
    "TRUE_FUTURE_LATENT_COST",
    "TWO_STEP_PREDICTED",
    "TWO_STEP_PREDICTED_LATENT_COST",
    "aggregate_state_metrics",
    "all_candidate_tendency_diagnostics",
    "canonical_json_bytes",
    "canonical_sha256",
    "classify_qualification",
    "classify_route_population",
    "deterministic_random_order",
    "evaluate_absolute_base_preservation_screen",
    "evaluate_predicted_gate",
    "evaluate_state_population",
    "evaluate_true_future_gate",
    "filter_population",
    "kinematic_nominal_outcome",
    "kinematic_rank_costs",
    "kinematic_route_order",
    "latent_progress_diagnostics",
    "layer_normalize_tokens",
    "margin_borda_utility",
    "next_experiment",
    "paired_family_state_bootstrap",
    "paired_state_bootstrap",
    "paired_source_comparison",
    "realised_route_order",
    "route_ordering_metrics",
    "route_preference",
    "summarize_source",
    "summarize_state_families",
    "tokenwise_normalized_cosine_costs",
    "tokenwise_normalized_cosine_mean_cost",
    "trajectory_cost_monotonicity",
    "type7_quantile",
]
