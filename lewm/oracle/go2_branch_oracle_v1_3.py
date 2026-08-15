"""Counterfactual branch oracle v1.3 -- graph-boundary outcomes.

This is a pure label contract.  It imports the frozen v1.2 progress, safety,
completion and utility implementations and adds one deliberately narrow rule:
when the *final* command-tick sample is outside the represented graph's frozen
2 m locate radius, or is located but cannot reach the designated goal under the
frozen transit mask, progress is the existing clipped lower bound ``-1.0``.

Intermediate graph-boundary samples are persisted diagnostics and do not latch.
Consequently every branch accepted by v1.2 has exactly the same legacy label
projection under v1.3.  Missing, malformed or numeric-invalid evidence is still
refused; this module never fabricates a distance for a graph-boundary sample.

The module contains no Genesis imports and performs no filesystem access.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Mapping, Sequence

from lewm.oracle import go2_branch_oracle_v1_2 as V12


ORACLE_NAME = "go2_branch_oracle_v1_3"
TRACE_SCHEMA = "go2_branch_oracle_v1_3_tick_trace_v1"

REACHABLE = "REACHABLE"
OUTSIDE_LOCATE_RADIUS = "OUTSIDE_LOCATE_RADIUS"
GOAL_UNREACHABLE = "GOAL_UNREACHABLE"
NUMERIC_INVALID = "NUMERIC_INVALID"
GRAPH_STATUSES = (
    REACHABLE,
    OUTSIDE_LOCATE_RADIUS,
    GOAL_UNREACHABLE,
    NUMERIC_INVALID,
)
GRAPH_BOUNDARY_STATUSES = frozenset({
    OUTSIDE_LOCATE_RADIUS,
    GOAL_UNREACHABLE,
})

LOCATE_MAX_DISTANCE_M = 2.0
HORIZON_TICKS = V12.HORIZON_BLOCKS * V12.TICKS_PER_BLOCK
BOUNDARY_PROGRESS = -1.0

# These fields are the exact aggregate surface emitted by score_branch_v12.
# The compatibility receipt compares this projection, not v1.3 diagnostics.
LEGACY_LABEL_KEYS = (
    "start_geodesic_m",
    "final_geodesic_m",
    "progress",
    "contact_fraction",
    "clearance_cost",
    "stuck_fraction",
    "fall",
    "safety",
    "completion",
    "utility",
    "min_clearance_m",
    "evaluation_points",
)


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _finite_nonnegative(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def classify_graph_status(
    locate_distance_m: float | None,
    geodesic_m: float | None,
    *,
    pose_finite: bool = True,
) -> str:
    """Classify one graph-label sample under the frozen precedence.

    ``geodesic_m=None`` and positive infinity both mean that a located cell is
    goal-unreachable.  JSON traces must nevertheless store the unavailable
    distance as ``null`` rather than a non-standard infinity token.
    """

    if pose_finite is not True or not _finite_nonnegative(locate_distance_m):
        return NUMERIC_INVALID
    if float(locate_distance_m) > LOCATE_MAX_DISTANCE_M:
        return OUTSIDE_LOCATE_RADIUS
    if geodesic_m is None:
        return GOAL_UNREACHABLE
    if isinstance(geodesic_m, bool) or not isinstance(geodesic_m, (int, float)):
        return NUMERIC_INVALID
    value = float(geodesic_m)
    if math.isnan(value) or value == -math.inf or value < 0.0:
        return NUMERIC_INVALID
    if value == math.inf:
        return GOAL_UNREACHABLE
    return REACHABLE


def _validate_graph_sample(row: Mapping[str, Any]) -> tuple[str, float | None] | None:
    status = row.get("graph_status")
    if status not in GRAPH_STATUSES or status == NUMERIC_INVALID:
        return None
    geodesic = row.get("geodesic_m")
    if status == REACHABLE:
        if not _finite_nonnegative(geodesic):
            return None
        return str(status), float(geodesic)
    # Boundary samples carry a reason, not an invented or non-JSON distance.
    if geodesic is not None:
        return None
    return str(status), None


def _validate_tick(row: Any) -> tuple[str, float | None] | None:
    if not isinstance(row, Mapping):
        return None
    graph = _validate_graph_sample(row)
    if graph is None:
        return None
    contacts = row.get("disallowed_contacts")
    if (
        isinstance(contacts, bool)
        or not isinstance(contacts, int)
        or contacts < 0
        or not _finite_nonnegative(row.get("clearance_m"))
        or not isinstance(row.get("stuck"), bool)
        or not isinstance(row.get("terminated"), bool)
        or not isinstance(row.get("at_goal_cell"), bool)
    ):
        return None
    return graph


def legacy_label_projection(score: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact v1.2-compatible aggregate label surface."""

    missing = [key for key in LEGACY_LABEL_KEYS if key not in score]
    if missing:
        raise ValueError(f"score lacks legacy label fields: {missing}")
    return {key: score[key] for key in LEGACY_LABEL_KEYS}


def score_branch_v13(branch: Mapping[str, Any]) -> dict[str, Any] | None:
    """Score one complete twenty-tick trace, or refuse malformed evidence.

    The input retains the v1.2 branch shape: ``start``, ``ticks`` and ``nan``.
    Each graph sample additionally carries ``graph_status``.  Tick safety and
    completion keys retain their v1.2 meanings, including completion based on
    nearest-cell equality without a locate-radius condition.
    """

    if not isinstance(branch, Mapping) or branch.get("nan") is not False:
        return None
    start = branch.get("start")
    ticks = branch.get("ticks")
    if (
        not isinstance(start, Mapping)
        or not isinstance(ticks, Sequence)
        or isinstance(ticks, (str, bytes, bytearray))
        or len(ticks) != HORIZON_TICKS
    ):
        return None
    start_graph = _validate_graph_sample(start)
    if start_graph is None or start_graph[0] != REACHABLE:
        return None
    start_m = start_graph[1]
    assert start_m is not None

    graph_rows: list[tuple[str, float | None]] = []
    safety_rows: list[V12.TickSafetyEvidence] = []
    for row in ticks:
        validated = _validate_tick(row)
        if validated is None:
            return None
        graph_rows.append(validated)
        assert isinstance(row, Mapping)
        safety_rows.append(V12.TickSafetyEvidence(
            disallowed_contact=int(row["disallowed_contacts"]) > 0,
            clearance_m=float(row["clearance_m"]),
            stuck=bool(row["stuck"]),
            terminated=bool(row["terminated"]),
        ))

    final_status, final_m = graph_rows[-1]
    if final_status == REACHABLE:
        assert final_m is not None
        progress = V12.progress_from_distances(start_m, final_m)
        if progress is None:
            return None
    elif final_status in GRAPH_BOUNDARY_STATUSES:
        progress = BOUNDARY_PROGRESS
        final_m = None
    else:  # guarded above; retained as a fail-closed future-proofing branch
        return None

    safety = V12.graded_safety(safety_rows)
    if safety is None:
        return None
    completion = 1.0 if any(
        bool(row["at_goal_cell"]) for row in ticks if isinstance(row, Mapping)
    ) else 0.0

    first_boundary: int | None = None
    last_finite_tick = -1
    last_finite_m = float(start_m)
    for index, (status, distance) in enumerate(graph_rows):
        if status in GRAPH_BOUNDARY_STATUSES and first_boundary is None:
            first_boundary = index
        if status == REACHABLE:
            assert distance is not None
            last_finite_tick = index
            last_finite_m = float(distance)
    counts = Counter(status for status, _distance in graph_rows)

    result = {
        "start_geodesic_m": float(start_m),
        "final_geodesic_m": None if final_m is None else float(final_m),
        "progress": float(progress),
        "contact_fraction": safety["contact_fraction"],
        "clearance_cost": safety["clearance_cost"],
        "stuck_fraction": safety["stuck_fraction"],
        "fall": safety["fall"],
        "safety": safety["safety"],
        "completion": completion,
        "utility": V12.composite_utility(
            float(progress), safety["safety"], completion),
        "min_clearance_m": min(float(row["clearance_m"]) for row in ticks),
        "evaluation_points": len(ticks),
        "oracle": ORACLE_NAME,
        "trace_schema": TRACE_SCHEMA,
        "final_graph_status": final_status,
        "graph_boundary_contract_applied": final_status in GRAPH_BOUNDARY_STATUSES,
        "graph_boundary_occurred": first_boundary is not None,
        "transient_graph_boundary_recovered": (
            first_boundary is not None and final_status == REACHABLE
        ),
        "first_graph_boundary_tick": first_boundary,
        "last_finite_geodesic_tick": last_finite_tick,
        "last_finite_geodesic_m": last_finite_m,
        "graph_status_counts": {
            status: int(counts.get(status, 0)) for status in GRAPH_STATUSES
        },
    }
    return result


def oracle_contract() -> dict[str, Any]:
    """Return the frozen semantic contract without source or runtime state."""

    return {
        "schema": "go2_branch_oracle_v1_3_contract_v1",
        "name": ORACLE_NAME,
        "supersedes_oracle_v1_2_digest": V12.oracle_digest(),
        "inherited_progress_digest": V12.progress_digest(),
        "inherited_safety_digest": V12.safety_digest(),
        "horizon_blocks": V12.HORIZON_BLOCKS,
        "ticks_per_block": V12.TICKS_PER_BLOCK,
        "horizon_ticks": HORIZON_TICKS,
        "locate_radius_m": LOCATE_MAX_DISTANCE_M,
        "located_inequality": "nearest_node_distance_m <= 2.0",
        "graph_status_precedence": [
            NUMERIC_INVALID,
            OUTSIDE_LOCATE_RADIUS,
            GOAL_UNREACHABLE,
            REACHABLE,
        ],
        "reachable_final": "delegate v1.2 aggregate labels exactly",
        "graph_boundary_final": {
            "statuses": sorted(GRAPH_BOUNDARY_STATUSES),
            "progress": BOUNDARY_PROGRESS,
            "final_geodesic_m": None,
            "physical_safety": "unchanged v1.2 tick aggregation",
            "completion": "unchanged v1.2 at-goal-cell-at-any-tick",
            "utility": "1.0*progress - 2.0*safety + 0.5*completion",
        },
        "transient_graph_status": "persisted diagnostic; non-latching",
        "numeric_or_missing_trace": "refuse without a label",
        "completion_locate_radius_condition": False,
        "utility_weights": dict(V12.UTILITY_WEIGHTS),
        "trace_schema": TRACE_SCHEMA,
        "legacy_label_keys": list(LEGACY_LABEL_KEYS),
    }


def oracle_digest() -> str:
    return _digest(oracle_contract())


__all__ = [
    "BOUNDARY_PROGRESS",
    "GOAL_UNREACHABLE",
    "GRAPH_BOUNDARY_STATUSES",
    "GRAPH_STATUSES",
    "HORIZON_TICKS",
    "LEGACY_LABEL_KEYS",
    "LOCATE_MAX_DISTANCE_M",
    "NUMERIC_INVALID",
    "ORACLE_NAME",
    "OUTSIDE_LOCATE_RADIUS",
    "REACHABLE",
    "TRACE_SCHEMA",
    "classify_graph_status",
    "legacy_label_projection",
    "oracle_contract",
    "oracle_digest",
    "score_branch_v13",
]
