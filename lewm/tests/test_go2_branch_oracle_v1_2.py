"""Contract tests for the v1.2 branch oracle (continuous progress, graded safety).

These are the tests named in the v1.2 specification.  They run against a plain
scene-graph stub — no Genesis, no scene corpus, no checkpoint.
"""
from __future__ import annotations

import math

import pytest

from lewm.oracle.go2_branch_oracle_v1_2 import (
    CLEARANCE_SAFE_M,
    PROGRESS_NORMALISER_M,
    GeodesicField,
    TickSafetyEvidence,
    clearance_deficit,
    composite_utility,
    disallowed_contact_present,
    graded_safety,
    oracle_digest,
    progress_digest,
    progress_from_distances,
    safety_digest,
)


class _Graph:
    """Minimal SceneGraph stub: cell centres plus an adjacency list."""

    def __init__(self, centres, edges):
        self._centres = {int(k): (float(v[0]), float(v[1])) for k, v in centres.items()}
        self._adjacency = {int(k): [] for k in self._centres}
        for a, b in edges:
            self._adjacency[int(a)].append(int(b))
            self._adjacency[int(b)].append(int(a))

    def cell_center(self, cell_id):
        return self._centres[int(cell_id)]

    def neighbors(self, cell_id):
        return tuple(self._adjacency.get(int(cell_id), ()))


# 0 --- 1 --- 2 --- 3   on a straight line, 1 m apart, goal at cell 3.
LINE = _Graph({0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0), 3: (3.0, 0.0)},
              [(0, 1), (1, 2), (2, 3)])


def test_cell_distances_are_metric_not_hop_counts():
    field = GeodesicField(LINE, goal_cell=3)
    assert field.cell_distance(3) == pytest.approx(0.0)
    assert field.cell_distance(2) == pytest.approx(1.0)
    assert field.cell_distance(0) == pytest.approx(3.0)


def test_progress_varies_continuously_within_one_bfs_cell():
    """The whole point of v1.2: motion inside a cell must move the metric."""

    field = GeodesicField(LINE, goal_cell=3)
    # Every sample below sits in cell 1 (nearest centre), i.e. one BFS cell.
    xs = [0.80, 0.90, 1.00, 1.10, 1.20]
    distances = [field.remaining_distance((x, 0.0), cell_id=1) for x in xs]
    assert all(math.isfinite(d) for d in distances)
    # Strictly decreasing as the robot advances toward the goal.
    assert all(b < a for a, b in zip(distances, distances[1:]))
    # Integer BFS would have given a single constant value for all five.
    assert len(set(round(d, 9) for d in distances)) == len(xs)
    # And the change is smooth, not a jump.
    steps = [a - b for a, b in zip(distances, distances[1:])]
    assert all(s == pytest.approx(0.10, abs=1e-9) for s in steps)


def test_moving_along_the_shortest_path_gives_positive_progress():
    field = GeodesicField(LINE, goal_cell=3)
    start = field.remaining_distance((0.0, 0.0), cell_id=0)
    final = field.remaining_distance((0.5, 0.0), cell_id=0)
    progress = progress_from_distances(start, final)
    assert progress is not None and progress > 0.0
    assert progress == pytest.approx(0.5 / PROGRESS_NORMALISER_M)


def test_moving_away_from_the_goal_gives_negative_progress():
    field = GeodesicField(LINE, goal_cell=3)
    start = field.remaining_distance((1.0, 0.0), cell_id=1)
    final = field.remaining_distance((0.7, 0.0), cell_id=1)
    progress = progress_from_distances(start, final)
    assert progress is not None and progress < 0.0


def test_progress_is_clipped_to_the_unit_interval():
    assert progress_from_distances(100.0, 0.0) == 1.0
    assert progress_from_distances(0.0, 100.0) == -1.0


def test_unreachable_goal_is_refused_not_given_an_arbitrary_value():
    disconnected = _Graph({0: (0.0, 0.0), 1: (1.0, 0.0), 9: (9.0, 0.0)},
                          [(0, 1)])
    field = GeodesicField(disconnected, goal_cell=9)
    assert field.remaining_distance((0.0, 0.0), cell_id=0) == math.inf
    # Refusal propagates: the branch is never scored with a stand-in number.
    assert progress_from_distances(math.inf, math.inf) is None
    assert progress_from_distances(math.inf, 2.0) is None


def test_blocked_cells_are_endpoints_but_are_not_transited():
    """Mirrors SceneGraph.bfs_distance(transit_blocked=...) semantics."""

    # 0-1-2-3 straight; 0-4-3 a detour through 4. Blocking 1 forces the detour.
    graph = _Graph({0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0), 3: (3.0, 0.0),
                    4: (1.5, 5.0)},
                   [(0, 1), (1, 2), (2, 3), (0, 4), (4, 3)])
    direct = GeodesicField(graph, goal_cell=3)
    assert direct.cell_distance(0) == pytest.approx(3.0)
    detoured = GeodesicField(graph, goal_cell=3, transit_blocked={1})
    assert detoured.cell_distance(0) > 3.0
    # The blocked cell itself remains reachable as an endpoint.
    assert math.isfinite(detoured.cell_distance(1))


def test_equal_length_paths_break_ties_deterministically():
    # Two symmetric one-hop routes from 0 to the goal 3, via 1 and via 2.
    graph = _Graph({0: (0.0, 0.0), 1: (1.0, 1.0), 2: (1.0, -1.0), 3: (2.0, 0.0)},
                   [(0, 1), (0, 2), (1, 3), (2, 3)])
    fields = [GeodesicField(graph, goal_cell=3) for _ in range(5)]
    picks = {f.next_node(0) for f in fields}
    assert picks == {1}, "ties must resolve to the lowest cell id, every time"
    values = {f.remaining_distance((0.0, 0.0), cell_id=0) for f in fields}
    assert len(values) == 1


# ------------------------------------------------------------------ safety ---
def _evidence(n, *, contact=False, clearance=1.0, stuck=False, terminated=False):
    return [TickSafetyEvidence(disallowed_contact=contact, clearance_m=clearance,
                               stuck=stuck, terminated=terminated) for _ in range(n)]


def test_ordinary_safe_trajectory_does_not_receive_safety_one():
    """The v1.1 failure mode: 96 % of valid branches scored 1.0."""

    result = graded_safety(_evidence(20, clearance=0.60))
    assert result["safety"] == pytest.approx(0.0)
    assert result["contact_fraction"] == 0.0
    assert result["clearance_cost"] == 0.0
    assert result["stuck_fraction"] == 0.0


def test_longer_or_more_frequent_contact_produces_a_larger_cost():
    def cost(n_contact):
        rows = _evidence(20, clearance=0.60)
        rows = ([TickSafetyEvidence(True, 0.60, False, False)] * n_contact
                + rows[n_contact:])
        return graded_safety(rows)["safety"]

    values = [cost(k) for k in (0, 2, 5, 10, 20)]
    assert all(b > a for a, b in zip(values, values[1:]))
    assert values[-1] == pytest.approx(1.0 / 3.0)


def test_decreasing_clearance_produces_a_graded_increase():
    values = [graded_safety(_evidence(20, clearance=c))["safety"]
              for c in (0.20, 0.12, 0.08, 0.04, 0.0)]
    assert values[0] == pytest.approx(0.0)
    assert all(b > a for a, b in zip(values[1:], values[2:]))
    assert values[-1] == pytest.approx(1.0 / 3.0)
    # The deficit itself is the production threshold, normalised and clipped.
    assert clearance_deficit(CLEARANCE_SAFE_M) == pytest.approx(0.0)
    assert clearance_deficit(0.0) == pytest.approx(1.0)
    assert clearance_deficit(CLEARANCE_SAFE_M / 2) == pytest.approx(0.5)


def test_a_fall_produces_safety_one():
    rows = _evidence(20, clearance=0.60)
    rows[7] = TickSafetyEvidence(False, 0.60, False, terminated=True)
    assert graded_safety(rows)["safety"] == pytest.approx(1.0)
    assert graded_safety(rows)["fall"] == 1.0


def test_safety_is_bounded_in_the_unit_interval():
    worst = graded_safety(_evidence(20, contact=True, clearance=0.0, stuck=True))
    assert worst["safety"] == pytest.approx(1.0)
    assert 0.0 <= worst["safety"] <= 1.0


def test_the_same_trace_always_produces_the_same_value():
    rows = _evidence(11, contact=False, clearance=0.09, stuck=True)
    rows[3] = TickSafetyEvidence(True, 0.02, False, False)
    values = {graded_safety(rows)["safety"] for _ in range(8)}
    assert len(values) == 1


def test_stuck_is_averaged_over_the_path_not_a_binary_event():
    """v1.1 penalised 'any stuck event occurred'; v1.2 must not."""

    rows = _evidence(20, clearance=0.60)
    rows[0] = TickSafetyEvidence(False, 0.60, stuck=True, terminated=False)
    one_event = graded_safety(rows)["safety"]
    assert one_event == pytest.approx((1.0 / 20.0) / 3.0)
    assert one_event < 0.02


def test_utility_weights_are_preserved():
    assert composite_utility(1.0, 0.0, 0.0) == pytest.approx(1.0)
    assert composite_utility(0.0, 1.0, 0.0) == pytest.approx(-2.0)
    assert composite_utility(0.0, 0.0, 1.0) == pytest.approx(0.5)


# ----------------------------------------------------------------- contact ---
_ROBOT = (15, 28)
_FEET = frozenset({24, 25, 26, 27})
_GROUND = frozenset({0})


def test_foot_ground_contact_is_allowed():
    contacts = {"link_a": [0, 0], "link_b": [24, 27]}
    assert disallowed_contact_present(
        contacts, robot_link_range=_ROBOT, foot_link_indices=_FEET,
        ground_link_indices=_GROUND) == 0


def test_body_against_wall_and_body_against_ground_are_disallowed():
    contacts = {"link_a": [15, 16, 24], "link_b": [11, 11, 11]}
    assert disallowed_contact_present(
        contacts, robot_link_range=_ROBOT, foot_link_indices=_FEET,
        ground_link_indices=_GROUND) == 3
    belly = {"link_a": [0], "link_b": [15]}
    assert disallowed_contact_present(
        belly, robot_link_range=_ROBOT, foot_link_indices=_FEET,
        ground_link_indices=_GROUND) == 1


def test_self_contacts_and_environment_only_contacts_are_ignored():
    contacts = {"link_a": [24, 3], "link_b": [16, 11]}
    assert disallowed_contact_present(
        contacts, robot_link_range=_ROBOT, foot_link_indices=_FEET,
        ground_link_indices=_GROUND) == 0


def test_sub_threshold_contact_forces_are_ignored():
    contacts = {"link_a": [15], "link_b": [11]}
    assert disallowed_contact_present(
        contacts, robot_link_range=_ROBOT, foot_link_indices=_FEET,
        ground_link_indices=_GROUND, forces=[0.0]) == 0
    assert disallowed_contact_present(
        contacts, robot_link_range=_ROBOT, foot_link_indices=_FEET,
        ground_link_indices=_GROUND, forces=[12.5]) == 1


def test_contract_digests_are_stable():
    assert progress_digest() == progress_digest()
    assert safety_digest() == safety_digest()
    assert len(oracle_digest()) == 64
