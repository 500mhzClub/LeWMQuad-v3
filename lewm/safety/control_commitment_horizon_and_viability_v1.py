"""Pure reducers for control-commitment horizon and viability analysis."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np


PHYSICS_DT_S = 0.002
COMMAND_DT_S = 0.1
PHYSICS_STEPS_PER_TICK = 50
HORIZONS = (1, 2, 3, 4, 5)
DISTANCE_TIE_M = 0.03
HEADING_TIE_RAD = math.radians(5.0)


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def wrap(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def integrate_prefix(post_slew: Sequence, waypoint: Sequence[float], ticks: int) -> np.ndarray:
    """Apply the frozen kinematic integration rule to a command prefix."""

    if ticks not in HORIZONS:
        raise ValueError("ticks must be in 1..5")
    commands = [tick for block in post_slew for tick in block][:ticks]
    if len(commands) != ticks:
        raise ValueError("command trace shorter than requested prefix")
    x = y = yaw = 0.0
    for vx, vy, wz in commands:
        x += (math.cos(yaw) * float(vx) - math.sin(yaw) * float(vy)) * COMMAND_DT_S
        y += (math.sin(yaw) * float(vx) + math.cos(yaw) * float(vy)) * COMMAND_DT_S
        yaw = wrap(yaw + float(wz) * COMMAND_DT_S)
    wx, wy, sin_goal, cos_goal = map(float, waypoint)
    goal_heading = math.atan2(sin_goal, cos_goal)
    progress = math.hypot(wx, wy) - math.hypot(wx - x, wy - y)
    heading = abs(wrap(goal_heading)) - abs(wrap(goal_heading - yaw))
    return np.asarray([x, y, math.sin(yaw), math.cos(yaw), progress, heading], np.float64)


def realised_prefix(
    start_pose: Sequence,
    waypoint_xy: Sequence[float],
    endpoint_pose: Sequence[float],
    route_heading_world_rad: float,
) -> dict:
    """Realised displacement, route progress, and heading improvement."""

    (x0, y0), yaw0, _z0 = start_pose
    x, y, yaw = map(float, endpoint_pose[:3])
    start_distance = math.hypot(float(waypoint_xy[0]) - x0, float(waypoint_xy[1]) - y0)
    end_distance = math.hypot(float(waypoint_xy[0]) - x, float(waypoint_xy[1]) - y)
    start_error = abs(wrap(float(route_heading_world_rad) - float(yaw0)))
    end_error = abs(wrap(float(route_heading_world_rad) - yaw))
    return {
        "displacement_m": float(math.hypot(x - float(x0), y - float(y0))),
        "progress_m": float(start_distance - end_distance),
        "heading_improvement_rad": float(start_error - end_error),
    }


def first_divergence_step(
    link_transforms: np.ndarray,
    *,
    position_tolerance_m: float = 1e-3,
    orientation_tolerance_rad: float = 1e-3,
) -> int | None:
    """First physics step where any candidate differs from candidate zero."""

    x = np.asarray(link_transforms, np.float64)
    if x.ndim != 4 or x.shape[-1] != 7:
        raise ValueError("expected [candidate, step, link, xyz+quaternion]")
    anchor = x[0]
    position = np.max(
        np.linalg.norm(x[..., :3] - anchor[None, ..., :3], axis=-1), axis=(0, 2)
    )
    dot = np.abs(np.sum(x[..., 3:] * anchor[None, ..., 3:], axis=-1))
    orientation = np.max(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)), axis=(0, 2))
    indices = np.flatnonzero(
        (position > position_tolerance_m) | (orientation > orientation_tolerance_rad)
    )
    return None if not len(indices) else int(indices[0])


def route_order(rows: Sequence[Mapping]) -> list[int]:
    """Frozen nominal distance, heading, candidate-index ordering."""

    remaining = list(range(len(rows)))
    ordered: list[int] = []
    while remaining:
        best_distance = max(float(rows[index]["nominal_progress_m"]) for index in remaining)
        near = [
            index
            for index in remaining
            if best_distance - float(rows[index]["nominal_progress_m"]) <= DISTANCE_TIE_M
        ]
        pick = min(
            near,
            key=lambda index: (
                -float(rows[index]["nominal_heading_improvement_rad"]),
                int(rows[index]["candidate_index"]),
            ),
        )
        ordered.append(pick)
        remaining.remove(pick)
    return ordered


def realised_preference(a: Mapping, b: Mapping) -> int:
    distance = float(a["realised_progress_m"]) - float(b["realised_progress_m"])
    if abs(distance) > DISTANCE_TIE_M:
        return 1 if distance > 0 else -1
    heading = float(a["realised_heading_improvement_rad"]) - float(
        b["realised_heading_improvement_rad"]
    )
    if abs(heading) > HEADING_TIE_RAD:
        return 1 if heading > 0 else -1
    return 0


def availability_class(
    rows: Sequence[Mapping], *, boundary_contact: bool, divergence_step: int | None
) -> str:
    if boundary_contact:
        return "PRE_EXISTING_CONTACT"
    safe = [row for row in rows if not bool(row["committed_contact"])]
    if safe:
        if any(float(row["realised_progress_m"]) > 0.0 for row in safe):
            return "SAFE_PROGRESS_ACTION_AVAILABLE"
        return "SAFE_NONPROGRESS_ACTION_AVAILABLE"
    first = [int(row["first_contact_step"]) for row in rows if row["first_contact_step"] is not None]
    if divergence_step is not None and first and max(first) <= divergence_step:
        return "CONTACT_PRECEDES_CANDIDATE_DIVERGENCE"
    return "NO_SAFE_ACTION_AVAILABLE"


def viability_class(
    *,
    boundary_contact: bool,
    safe_counts: Mapping[int, int],
    first_contact_steps: Sequence[int | None],
    divergence_step: int | None,
    shorter_horizon_technically_available: bool,
) -> str:
    """Classify a five-tick no-safe state without conflating causal cases."""

    if boundary_contact:
        return "PRE_EXISTING_CONTACT"
    finite = [int(step) for step in first_contact_steps if step is not None]
    if finite and divergence_step is not None and max(finite) <= divergence_step:
        return "CONTACT_BEFORE_CONTROL_AUTHORITY"
    if int(safe_counts.get(1, 0)) == 0:
        return "ONE_TICK_VIABILITY_FAILURE"
    if shorter_horizon_technically_available and any(
        int(safe_counts.get(horizon, 0)) > 0 for horizon in HORIZONS[:-1]
    ):
        return "COMMITMENT_HORIZON_FAILURE"
    if any(int(safe_counts.get(horizon, 0)) > 0 for horizon in HORIZONS[:-1]):
        return "UNRESOLVED"
    return "CANDIDATE_BANK_COVERAGE_FAILURE"


def fixture_payload() -> dict:
    post_slew = [[[0.2, 0.0, 0.1]] * 5]
    waypoint = [1.0, 0.0, 0.0, 1.0]
    nominal = integrate_prefix(post_slew, waypoint, 1)
    links = np.zeros((2, 5, 1, 7), np.float64)
    links[..., 3] = 1.0
    links[1, 3:, 0, 0] = 0.002
    rows = [
        {"candidate_index": 0, "nominal_progress_m": 0.2, "nominal_heading_improvement_rad": 0.0,
         "realised_progress_m": 0.1, "realised_heading_improvement_rad": 0.0, "committed_contact": False},
        {"candidate_index": 1, "nominal_progress_m": 0.1, "nominal_heading_improvement_rad": 0.1,
         "realised_progress_m": 0.2, "realised_heading_improvement_rad": 0.0, "committed_contact": True,
         "first_contact_step": 2},
    ]
    tests = {
        "one_tick_integrates": abs(float(nominal[0]) - 0.02) < 1e-12,
        "divergence_at_three": first_divergence_step(links) == 3,
        "ranker_prefers_nominal_distance": route_order(rows)[0] == 0,
        "safe_progress_class": availability_class(rows, boundary_contact=False, divergence_step=3)
        == "SAFE_PROGRESS_ACTION_AVAILABLE",
        "one_tick_failure_distinct": viability_class(
            boundary_contact=False,
            safe_counts={1: 0, 5: 0},
            first_contact_steps=[5, 6],
            divergence_step=2,
            shorter_horizon_technically_available=True,
        ) == "ONE_TICK_VIABILITY_FAILURE",
        "unavailable_shorter_horizon_is_unresolved": viability_class(
            boundary_contact=False,
            safe_counts={1: 1, 5: 0},
            first_contact_steps=[100, 110],
            divergence_step=2,
            shorter_horizon_technically_available=False,
        ) == "UNRESOLVED",
    }
    payload = {
        "schema": "control_commitment_horizon_and_viability_fixture_v1",
        "tests": tests,
        "pass": all(tests.values()),
    }
    payload["content_digest"] = digest(payload)
    return payload
