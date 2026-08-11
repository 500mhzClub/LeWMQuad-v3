"""Counterfactual branch oracle v1.2 — continuous progress, graded safety.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

v1.1 (`03dbe011…`) is frozen as a **failed** development oracle: integer BFS
progress was too coarse (ΔBFS was 0 in 162 of 203 valid branches), and taking the
max of binary contact / stuck / termination signals marked 96 % of valid branches
unsafe, collapsing the composite utility onto six values.

v1.2 changes exactly two things and nothing else:

* **progress** becomes a continuous metric geodesic quantity over the existing
  scene graph, normalised by the distance the fastest frozen candidate could
  cover in the frozen horizon;
* **safety** becomes a graded path cost in ``[0, 1]`` built from the *fraction*
  of evaluation points in disallowed contact, the *mean normalised clearance
  deficit*, and the *fraction* of evaluation points satisfying the production
  stuck predicate, with a fall dominating via an outer max.

Preserved unchanged: the twelve-candidate bank, the four-block / two-second
horizon, the snapshot and restoration implementation, the CPU backend,
snapshot-time landmark binding, at-or-before-horizon completion, the utility
weights ``U = 1.0·P − 2.0·S + 0.5·C``, the tie tolerance, and every gate
threshold.

This module is deliberately free of Genesis imports so the contracts can be unit
tested against a plain scene-graph stub.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

# ---- frozen horizon and candidate-derived normalisation -----------------------
HORIZON_BLOCKS = 4
TICKS_PER_BLOCK = 5
HORIZON_S = 2.0
# The largest translational set-point in the frozen bank 85471e44… (forward_fast).
V_MAX_MPS = 0.30
PROGRESS_NORMALISER_M = V_MAX_MPS * HORIZON_S      # 0.60 m

# ---- production safety-clearance threshold (unchanged from design v1.1) -------
CLEARANCE_SAFE_M = 0.15
# Production contact-force threshold, from ``RolloutRunner._extract_foot_contacts``.
CONTACT_FORCE_THRESHOLD_N = 1e-3

UTILITY_WEIGHTS = {"progress": 1.0, "safety": -2.0, "completion": 0.5}

PROGRESS_CONTRACT = {
    "name": "go2_branch_progress_v1_2",
    "quantity": "continuous metric geodesic remaining distance to the "
                "snapshot-bound landmark",
    "graph": "lewm_worlds.scene_graph.SceneGraph cell centres, edge weight = "
             "Euclidean distance between adjacent cell centres",
    "transit_semantics": "mirrors SceneGraph.bfs_distance(transit_blocked=...): a "
                         "blocked cell may be a path endpoint but is never "
                         "transited; the robot's own cell is exempt",
    "continuous_term": "distance from the robot's world xy to the centre of the "
                       "next node on the deterministic shortest path, added to "
                       "that node's Dijkstra distance",
    "tie_break": "argmin over (edge_weight + goal_distance, neighbour_cell_id) — "
                 "equal-length paths resolve to the lowest cell id",
    "unreachable": "returns infinity; the branch is refused, never scored",
    "formula": "P = clip((d_start - d_final) / (V_MAX_MPS * HORIZON_S), -1, 1)",
    "v_max_mps": V_MAX_MPS,
    "horizon_s": HORIZON_S,
    "normaliser_m": PROGRESS_NORMALISER_M,
}

SAFETY_CONTRACT = {
    "name": "go2_branch_safety_v1_2",
    "formula": "S = max(F, (C_contact + C_clearance + C_stuck) / 3)",
    "range": [0.0, 1.0],
    "evaluation_points": "one per command tick of the branch "
                         f"({HORIZON_BLOCKS * TICKS_PER_BLOCK} points, 10 Hz, the "
                         "production emission cadence)",
    "F": "1.0 on fall or unsafe termination (production RolloutRunner reset "
         "predicates: fall-z, out-of-bounds, tip), else 0.0",
    "C_contact": {
        "definition": "fraction of evaluation points with at least one actual "
                      "disallowed robot/environment contact",
        "source": "genesis RigidEntity.get_contacts() from the executed branch",
        "allowed": "ordinary foot-ground contact only — a robot calf link "
                   "against the ground plane",
        "disallowed": "any robot link against non-ground environment geometry, "
                      "and any non-calf robot link against the ground",
        "excluded": "robot self-contacts (not robot/environment)",
        "force_threshold_n": CONTACT_FORCE_THRESHOLD_N,
        "not_used": "_body_probe_configuration_clearance_m — the v1.1 pilot "
                    "showed its 0.35 x 0.18 m footprint labels ordinary maze "
                    "corridors as contact",
    },
    "C_clearance": {
        "definition": "mean over evaluation points of "
                      "clip((c_safe - c_t) / c_safe, 0, 1)",
        "c_safe_m": CLEARANCE_SAFE_M,
        "source": "lewm_worlds.scene_graph.SceneGraph.clearance_to_walls",
    },
    "C_stuck": {
        "definition": "fraction of evaluation points satisfying the production "
                      "stuck predicate, averaged over the path — never 'any "
                      "stuck event occurred'",
        "source": "lewm_worlds.labels.derived.DerivedLabelComputer.stuck_label",
    },
}

COMPLETION_CONTRACT = {
    "name": "go2_branch_completion_v1_2",
    "definition": "the bound landmark's cell is reached at or before the branch "
                  "horizon (unchanged from v1.1)",
}


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def progress_digest() -> str:
    return _digest(PROGRESS_CONTRACT)


def safety_digest() -> str:
    return _digest(SAFETY_CONTRACT)


def oracle_digest() -> str:
    return _digest({
        "progress": PROGRESS_CONTRACT,
        "safety": SAFETY_CONTRACT,
        "completion": COMPLETION_CONTRACT,
        "weights": UTILITY_WEIGHTS,
        "horizon_blocks": HORIZON_BLOCKS,
        "ticks_per_block": TICKS_PER_BLOCK,
        "supersedes": "03dbe01100870cb4cc082f936bc3d0b62aa1e5d23e8eedb44188e88992acbb53",
    })


# --------------------------------------------------------------- geodesic ----
class GeodesicField:
    """Metric remaining-distance to one landmark cell.

    Dijkstra runs *from* the goal over the traversable subgraph with Euclidean
    edge weights, so one field serves every query during a branch.  The
    ``transit_blocked`` semantics are the reverse-direction mirror of
    :meth:`SceneGraph.bfs_distance`: a blocked cell can be an endpoint but its
    neighbours are never expanded through it, and the goal itself is exempt.
    """

    __slots__ = ("goal_cell", "_distance", "_graph", "_blocked")

    def __init__(self, graph: Any, goal_cell: int,
                 transit_blocked: Iterable[int] | None = None) -> None:
        self.goal_cell = int(goal_cell)
        self._graph = graph
        self._blocked = frozenset(int(c) for c in (transit_blocked or ()))
        self._distance = self._dijkstra()

    def _edge_weight(self, a: int, b: int) -> float:
        ax, ay = self._graph.cell_center(int(a))
        bx, by = self._graph.cell_center(int(b))
        return math.hypot(ax - bx, ay - by)

    def _dijkstra(self) -> dict[int, float]:
        distance: dict[int, float] = {self.goal_cell: 0.0}
        # (dist, cell) ordering makes the frontier order deterministic.
        queue: list[tuple[float, int]] = [(0.0, self.goal_cell)]
        settled: set[int] = set()
        while queue:
            best, node = heapq.heappop(queue)
            if node in settled:
                continue
            settled.add(node)
            # A blocked cell is reachable as an endpoint but is not transited.
            if node != self.goal_cell and node in self._blocked:
                continue
            for neighbour in self._graph.neighbors(node):
                neighbour = int(neighbour)
                if neighbour in settled:
                    continue
                candidate = best + self._edge_weight(node, neighbour)
                if candidate < distance.get(neighbour, math.inf):
                    distance[neighbour] = candidate
                    heapq.heappush(queue, (candidate, neighbour))
        return distance

    def cell_distance(self, cell_id: int) -> float:
        return float(self._distance.get(int(cell_id), math.inf))

    def next_node(self, cell_id: int) -> int | None:
        """The next node on the deterministic shortest path out of ``cell_id``."""

        cell_id = int(cell_id)
        if cell_id == self.goal_cell:
            return None
        best: tuple[float, int] | None = None
        for neighbour in self._graph.neighbors(cell_id):
            neighbour = int(neighbour)
            goal_distance = self.cell_distance(neighbour)
            if not math.isfinite(goal_distance):
                continue
            key = (self._edge_weight(cell_id, neighbour) + goal_distance, neighbour)
            if best is None or key < best:
                best = key
        return None if best is None else best[1]

    def remaining_distance(self, xy: Sequence[float], cell_id: int) -> float:
        """Continuous metres remaining, or ``inf`` when the goal is unreachable.

        Varies continuously with ``xy`` *within* a cell — it is not a step
        function of the integer cell index.
        """

        cell_id = int(cell_id)
        if not math.isfinite(self.cell_distance(cell_id)):
            return math.inf
        x, y = float(xy[0]), float(xy[1])
        if cell_id == self.goal_cell:
            gx, gy = self._graph.cell_center(self.goal_cell)
            return math.hypot(x - gx, y - gy)
        nxt = self.next_node(cell_id)
        if nxt is None:
            return math.inf
        nx, ny = self._graph.cell_center(nxt)
        return math.hypot(x - nx, y - ny) + self.cell_distance(nxt)


def progress_from_distances(start_m: float, final_m: float) -> float | None:
    """``P`` under the frozen v1.2 formula, or ``None`` when either end is unreachable."""

    if not (math.isfinite(start_m) and math.isfinite(final_m)):
        return None
    raw = (float(start_m) - float(final_m)) / PROGRESS_NORMALISER_M
    return float(max(-1.0, min(1.0, raw)))


# ----------------------------------------------------------------- safety ----
@dataclass(frozen=True)
class TickSafetyEvidence:
    """One evaluation point of the graded path cost."""

    disallowed_contact: bool
    clearance_m: float
    stuck: bool
    terminated: bool


def clearance_deficit(clearance_m: float) -> float:
    """``clip((c_safe - c_t) / c_safe, 0, 1)`` — 0 when clear, 1 at the wall."""

    deficit = (CLEARANCE_SAFE_M - float(clearance_m)) / CLEARANCE_SAFE_M
    return float(max(0.0, min(1.0, deficit)))


def graded_safety(evidence: Sequence[TickSafetyEvidence]) -> dict[str, float] | None:
    """``S = max(F, (C_contact + C_clearance + C_stuck) / 3)`` in ``[0, 1]``."""

    if not evidence:
        return None
    n = float(len(evidence))
    contact = sum(1.0 for row in evidence if row.disallowed_contact) / n
    clearance = sum(clearance_deficit(row.clearance_m) for row in evidence) / n
    stuck = sum(1.0 for row in evidence if row.stuck) / n
    fall = 1.0 if any(row.terminated for row in evidence) else 0.0
    graded = (contact + clearance + stuck) / 3.0
    total = max(fall, graded)
    return {
        "contact_fraction": float(contact),
        "clearance_cost": float(clearance),
        "stuck_fraction": float(stuck),
        "fall": float(fall),
        "graded_mean": float(graded),
        "safety": float(max(0.0, min(1.0, total))),
    }


# ---------------------------------------------------------------- utility ----
def composite_utility(progress: float, safety: float, completion: float) -> float:
    return float(UTILITY_WEIGHTS["progress"] * progress
                 + UTILITY_WEIGHTS["safety"] * safety
                 + UTILITY_WEIGHTS["completion"] * completion)


# ------------------------------------------------------- contact classifier --
def disallowed_contact_present(contacts: Mapping[str, Any], *,
                               robot_link_range: tuple[int, int],
                               foot_link_indices: frozenset[int],
                               ground_link_indices: frozenset[int],
                               forces: Sequence[float] | None = None) -> int:
    """Count actual disallowed robot/environment contacts in one query.

    Allowed: a robot calf (the Go2 URDF's terminal foot feature) against the
    ground plane.  Everything else that pairs a robot link with environment
    geometry is disallowed.  Robot self-contacts are excluded — they are not
    robot/environment contact.
    """

    link_a = [int(v) for v in contacts.get("link_a", ())]
    link_b = [int(v) for v in contacts.get("link_b", ())]
    low, high = robot_link_range
    count = 0
    for index, (a, b) in enumerate(zip(link_a, link_b)):
        a_robot = low <= a < high
        b_robot = low <= b < high
        if a_robot == b_robot:
            continue                       # self-contact, or neither side is the robot
        robot_link = a if a_robot else b
        other_link = b if a_robot else a
        if forces is not None and index < len(forces):
            if float(forces[index]) <= CONTACT_FORCE_THRESHOLD_N:
                continue
        on_ground = other_link in ground_link_indices
        if on_ground and robot_link in foot_link_indices:
            continue                       # ordinary foot-ground contact
        count += 1
    return count
