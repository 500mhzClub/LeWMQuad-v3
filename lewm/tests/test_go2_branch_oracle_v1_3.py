"""Pure synthetic tests for the frozen oracle-v1.3 boundary contract."""
from __future__ import annotations

import copy
import math

import pytest

from lewm.oracle import go2_branch_oracle_v1_2 as V12
from lewm.oracle import go2_branch_oracle_v1_3 as V13


def _tick(
    *,
    status: str = V13.REACHABLE,
    geodesic_m: float | None = 2.0,
    clearance_m: float = 0.6,
    contacts: int = 0,
    stuck: bool = False,
    terminated: bool = False,
    at_goal: bool = False,
) -> dict:
    return {
        "graph_status": status,
        "geodesic_m": geodesic_m,
        "disallowed_contacts": contacts,
        "clearance_m": clearance_m,
        "stuck": stuck,
        "terminated": terminated,
        "at_goal_cell": at_goal,
    }


def _branch(*, start_m: float = 2.4, final_m: float = 2.0) -> dict:
    ticks = [_tick(geodesic_m=start_m - 0.01 * (index + 1))
             for index in range(V13.HORIZON_TICKS)]
    ticks[-1]["geodesic_m"] = final_m
    return {
        "nan": False,
        "start": {"graph_status": V13.REACHABLE, "geodesic_m": start_m},
        "ticks": ticks,
    }


def _v12_projection(branch: dict) -> dict:
    start_m = float(branch["start"]["geodesic_m"])
    final_m = float(branch["ticks"][-1]["geodesic_m"])
    progress = V12.progress_from_distances(start_m, final_m)
    assert progress is not None
    evidence = [V12.TickSafetyEvidence(
        disallowed_contact=int(row["disallowed_contacts"]) > 0,
        clearance_m=float(row["clearance_m"]),
        stuck=bool(row["stuck"]),
        terminated=bool(row["terminated"]),
    ) for row in branch["ticks"]]
    safety = V12.graded_safety(evidence)
    assert safety is not None
    completion = 1.0 if any(row["at_goal_cell"] for row in branch["ticks"]) else 0.0
    return {
        "start_geodesic_m": start_m,
        "final_geodesic_m": final_m,
        "progress": float(progress),
        "contact_fraction": safety["contact_fraction"],
        "clearance_cost": safety["clearance_cost"],
        "stuck_fraction": safety["stuck_fraction"],
        "fall": safety["fall"],
        "safety": safety["safety"],
        "completion": completion,
        "utility": V12.composite_utility(progress, safety["safety"], completion),
        "min_clearance_m": min(float(row["clearance_m"])
                               for row in branch["ticks"]),
        "evaluation_points": len(branch["ticks"]),
    }


def test_graph_status_boundary_is_strictly_above_two_metres():
    assert V13.classify_graph_status(2.0, 3.0) == V13.REACHABLE
    assert V13.classify_graph_status(
        math.nextafter(2.0, math.inf), None
    ) == V13.OUTSIDE_LOCATE_RADIUS
    assert V13.classify_graph_status(0.1, math.inf) == V13.GOAL_UNREACHABLE
    assert V13.classify_graph_status(0.1, None) == V13.GOAL_UNREACHABLE


@pytest.mark.parametrize(
    "locate,geodesic,pose_finite",
    [
        (math.nan, 1.0, True),
        (math.inf, 1.0, True),
        (-0.1, 1.0, True),
        (0.1, math.nan, True),
        (0.1, -1.0, True),
        (0.1, 1.0, False),
    ],
)
def test_numeric_invalid_precedes_scientific_boundary_status(
    locate, geodesic, pose_finite,
):
    assert V13.classify_graph_status(
        locate, geodesic, pose_finite=pose_finite
    ) == V13.NUMERIC_INVALID


def test_reachable_final_has_exact_v12_label_projection():
    branch = _branch()
    branch["ticks"][3].update(
        disallowed_contacts=1, clearance_m=0.04, stuck=True
    )
    branch["ticks"][11]["at_goal_cell"] = True
    score = V13.score_branch_v13(branch)
    assert score is not None
    assert V13.legacy_label_projection(score) == _v12_projection(branch)
    assert score["graph_boundary_contract_applied"] is False


@pytest.mark.parametrize(
    "status",
    [V13.OUTSIDE_LOCATE_RADIUS, V13.GOAL_UNREACHABLE],
)
def test_final_graph_boundary_uses_progress_floor_and_physical_safety(status):
    branch = _branch()
    branch["ticks"][-1] = _tick(status=status, geodesic_m=None)
    score = V13.score_branch_v13(branch)
    assert score is not None
    assert score["progress"] == -1.0
    assert score["final_geodesic_m"] is None
    assert score["safety"] == 0.0
    assert score["completion"] == 0.0
    assert score["utility"] == -1.0
    assert score["final_graph_status"] == status
    assert score["graph_boundary_contract_applied"] is True


def test_transient_boundary_is_diagnostic_and_does_not_latch():
    reference = _branch()
    transient = copy.deepcopy(reference)
    transient["ticks"][4] = _tick(
        status=V13.OUTSIDE_LOCATE_RADIUS, geodesic_m=None
    )
    expected = V13.score_branch_v13(reference)
    actual = V13.score_branch_v13(transient)
    assert expected is not None and actual is not None
    assert actual["progress"] == expected["progress"]
    assert actual["final_graph_status"] == V13.REACHABLE
    assert actual["first_graph_boundary_tick"] == 4
    assert actual["transient_graph_boundary_recovered"] is True


def test_completion_keeps_v12_nearest_cell_semantics_during_boundary_tick():
    branch = _branch()
    branch["ticks"][2] = _tick(
        status=V13.GOAL_UNREACHABLE, geodesic_m=None, at_goal=True
    )
    score = V13.score_branch_v13(branch)
    assert score is not None
    assert score["completion"] == 1.0
    assert score["progress"] == V12.progress_from_distances(2.4, 2.0)


@pytest.mark.parametrize("mutation", ["short", "nan", "numeric", "missing"])
def test_missing_or_numeric_invalid_trace_is_refused(mutation):
    branch = _branch()
    if mutation == "short":
        branch["ticks"].pop()
    elif mutation == "nan":
        branch["nan"] = True
    elif mutation == "numeric":
        branch["ticks"][7]["graph_status"] = V13.NUMERIC_INVALID
    else:
        del branch["ticks"][7]["clearance_m"]
    assert V13.score_branch_v13(branch) is None


def test_boundary_trace_uses_null_not_infinity():
    branch = _branch()
    branch["ticks"][-1].update(
        graph_status=V13.GOAL_UNREACHABLE,
        geodesic_m=math.inf,
    )
    assert V13.score_branch_v13(branch) is None


def test_oracle_digest_is_deterministic_and_binds_endpoint_policy():
    assert V13.oracle_digest() == V13.oracle_digest()
    assert len(V13.oracle_digest()) == 64
    contract = V13.oracle_contract()
    assert contract["graph_boundary_final"]["progress"] == -1.0
    assert contract["transient_graph_status"].endswith("non-latching")
