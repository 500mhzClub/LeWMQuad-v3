"""TopologicalNavigator — the learned Memory for the HierarchicalPlanner seam.

Stage 3 wiring (build plan §5 Stage 3): wraps ``OnlineTopologicalMemory``
(view-keyframe nodes + §5.4 top-k Bayes filter, validated offline at 0.96
trajectory coherence) behind the abstract ``Memory`` contract from Stage 0, and
adds the two Level-1/2 pieces:

  - **Goal matching = raw-frame cosine** between the goal frame's frozen latent
    and each node's *representative observation* latent (stored at commit).
    Evidence-driven: the GoalAdapter gate showed belief space is no better
    than the raw heading-dominated code at view-level matching (oracle view-R@1
    0.297 vs raw 0.301), and nodes are view keyframes — so the frozen frame
    code is the strongest goal->node matcher available, with zero parameters.
    (Third instance of the repo pattern: plan_cost > GoalEnergyHead, v6 > v7
    yaw-objective, raw-frame > GoalAdapter.)
  - **Routing = BFS over memory edges** (treated as undirected; the platform
    can reverse). No learned ReachabilityHead: A3 already showed latent-pair
    reachability generalizes at baseline, and the spec's own principle #4 says
    routing is graph BFS over the learned memory. The head returns only if
    routing proves graph-incompleteness-limited.

Per-update flow: maintain the H-frame frozen-latent window; once full, embed
with the (frozen) BeliefEncoder and step the filter. Node commits record the
current frame's raw latent (+ optional opaque observation reference, e.g. the
image/path, for handing to LocalMPC as a sub-goal image).

``select_subgoal``: match the goal against node keyframes; if confident and a
path exists from the MAP node, return the next node's representative
observation as the sub-goal (goal-facing by construction — view keyframes).
Otherwise fall back to the original goal (v2 behaviour; exploration mode is
Stage 4 work).
"""
from __future__ import annotations

from collections import deque
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from lewm.memory.online_topological_memory import MemoryNode, OnlineTopologicalMemory
from lewm.memory.topological_memory import Memory
from lewm.planning.local_mpc import GoalSpec


class TopologicalNavigator(Memory):
    def __init__(
        self,
        belief_encoder,                       # frozen BeliefEncoder (eval mode)
        pair_scorer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        history: int = 8,
        tau_new: float,
        tau_goal: float = 0.80,
        new_node_streak: int = 3,
        top_k: int = 8,
        self_stay_prob: float = 0.6,
        uniform_leak: float = 0.05,
        subgoal_lookahead: int = 4,
    ) -> None:
        self.belief_encoder = belief_encoder
        self.history = int(history)
        self.tau_goal = float(tau_goal)
        self.subgoal_lookahead = int(subgoal_lookahead)
        self.memory = OnlineTopologicalMemory(
            pair_scorer, tau_new=tau_new, new_node_streak=new_node_streak,
            top_k=top_k, self_stay_prob=self_stay_prob, uniform_leak=uniform_leak,
        )
        self._window: deque[torch.Tensor] = deque(maxlen=self.history)
        self._keyframes: dict[int, torch.Tensor] = {}     # node_id -> raw frame latent (L2-normed)
        self._observations: dict[int, Any] = {}           # node_id -> opaque obs reference
        # Mapping-time IMU yaw for directed traversals. Normal walked edges and
        # synthetic terminal spurs both use this deployment-admissible heading
        # instead of repeatedly recovering direction from a visual scan.
        self._edge_bearings: dict[tuple[int, int], float] = {}
        self._last_map: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Memory contract
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def update(self, observation: Any, action_block: Optional[Any] = None, *, label=None) -> Optional[int]:
        """``observation`` = (z_raw frame latent (D,), opaque obs reference)."""
        z_raw, obs_ref = observation if isinstance(observation, tuple) else (observation, None)
        self._window.append(z_raw)
        if len(self._window) < self.history:
            return None
        window = torch.stack(list(self._window)).unsqueeze(0)
        belief = self.belief_encoder(window).squeeze(0)
        n_before = len(self.memory.nodes)
        map_node = self.memory.update(belief, label=label)
        if len(self.memory.nodes) > n_before:                 # a node was committed
            node_id = self.memory.nodes[-1].node_id
            self._keyframes[node_id] = F.normalize(z_raw, dim=-1)
            self._observations[node_id] = obs_ref
        self._last_map = map_node
        return map_node

    @torch.no_grad()
    def insert_spur(
        self, z_raw: torch.Tensor, obs_ref: Any, anchor: int, *,
        label=None, bearing_rad: Optional[float] = None,
    ) -> int:
        """Register a look-aside view (e.g. a landmark standoff photo taken by
        stepping off the path and returning) as a TERMINAL SPUR node: a node
        with anchor<->spur edges that does not touch the filter window,
        posterior, or transition chain. Feeding such frames through ``update``
        instead either splices the spur INLINE into the locomotion chain —
        every passing route then detours into the standoff (measured: the
        wide-maze v18/v19 seeks collapsed into veto/realign storms at the
        first en-route beacon) — or mints duplicate chain nodes, because the
        history-dependent belief embedding cannot loop-close from a few
        stationary frames. The spur embedding is the BeliefEncoder over a
        stationary window of the view: what the live window converges to when
        the robot stands facing the landmark at arrival. ``bearing_rad`` records
        the mapping-time IMU yaw used to traverse the synthetic anchor->spur
        edge; unlike a normal walked edge, that direction cannot be recovered by
        matching the close-up spur view while still standing at the anchor."""
        window = z_raw.unsqueeze(0).repeat(self.history, 1).unsqueeze(0)
        embedding = F.normalize(self.belief_encoder(window).squeeze(0), dim=-1)
        node = MemoryNode(node_id=len(self.memory.nodes), embedding=embedding, in_filter=False)
        if label is not None:
            node.member_labels.append(label)
        self.memory.nodes.append(node)
        for key in ((int(anchor), node.node_id), (node.node_id, int(anchor))):
            self.memory.edges[key] = self.memory.edges.get(key, 0) + 1
        if bearing_rad is not None:
            self._edge_bearings[(int(anchor), node.node_id)] = float(bearing_rad)
        self._keyframes[node.node_id] = F.normalize(z_raw, dim=-1)
        self._observations[node.node_id] = obs_ref
        return node.node_id

    def current_belief(self) -> dict[int, float]:
        return dict(self.memory.posterior)

    @torch.no_grad()
    def select_subgoal(self, goal: GoalSpec) -> GoalSpec:
        plan = self.plan_to_goal_latent(goal.goal_image, lookahead=self.subgoal_lookahead)
        if plan is None:
            return goal                                       # fallback: v2 behaviour
        next_node, goal_node, _score = plan
        return GoalSpec(goal_image=self._observations.get(next_node), subgoal_node_id=next_node)

    # ------------------------------------------------------------------ #
    # Level-1 internals (also used directly by the offline routing probe)
    # ------------------------------------------------------------------ #

    def match_goal(self, goal_latent: torch.Tensor) -> tuple[Optional[int], float]:
        """Raw-frame cosine against node keyframes -> (best node, score)."""
        if not self._keyframes:
            return None, 0.0
        node_ids = sorted(self._keyframes)
        keyframes = torch.stack([self._keyframes[i] for i in node_ids])
        scores = keyframes @ F.normalize(goal_latent, dim=-1)
        best = int(scores.argmax())
        return node_ids[best], float(scores[best])

    def bfs_path(self, start: int, target: int, *, directed: bool = False) -> Optional[list[int]]:
        """Shortest node path start->target incl. both ends (None if disconnected).

        ``directed=True`` follows edges only in their recorded direction — the
        Stage 4b traversal mode: a directed edge A->B guarantees B's keyframe
        faces the A->B travel direction (the tour walked into B facing it), so
        align-to-keyframe + walk-forward traverses it without any metric cost.
        """
        if start == target:
            return [start]
        adjacency: dict[int, set[int]] = {}
        for (a, b) in self.memory.edges:
            adjacency.setdefault(a, set()).add(b)
            if not directed:
                adjacency.setdefault(b, set()).add(a)
        frontier, parent = [start], {start: None}
        while frontier and target not in parent:
            nxt = []
            for node in frontier:
                for neighbor in adjacency.get(node, ()):
                    if neighbor not in parent:
                        parent[neighbor] = node
                        nxt.append(neighbor)
            frontier = nxt
        if target not in parent:
            return None
        path = [target]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        return path[::-1]

    def bfs_next_hop(self, start: int, target: int, *, lookahead: int = 1) -> Optional[int]:
        """Sub-goal node ``lookahead`` steps along the path (clamped to target).

        ``lookahead`` exists because view-keyframe fragmentation (~4 nodes per
        place) makes the literal first hop usually another view of the *same*
        place — the offline routing probe measured progress 0.18 vs random 0.47
        at lookahead=1. Skipping a few nodes ahead steps over the same-place
        cluster while staying within local-servoing range.
        """
        path = self.bfs_path(start, target)
        if path is None:
            return None
        return path[min(max(int(lookahead), 1), len(path) - 1)]

    def adaptive_next_hop(self, start: int, target: int, *, tau_place: float = 0.8) -> Optional[int]:
        """First path node OUTSIDE the start's place cluster, detected in
        BELIEF space (deployment-valid: v6's any-yaw same-cell positives make
        belief similarity place-like/yaw-blurred, while keyframes are
        view-like). Skips the same-place view cluster that defeats lookahead=1
        without the fixed-k guess."""
        path = self.bfs_path(start, target)
        if path is None:
            return None
        start_embedding = self.memory.nodes[start].embedding
        for node_id in path[1:]:
            similarity = float(self.memory.nodes[node_id].embedding @ start_embedding)
            if similarity < tau_place:
                return node_id
        return path[-1]

    def plan_to_goal_latent(
        self, goal_latent: torch.Tensor, *, lookahead: int = 1
    ) -> Optional[tuple[int, int, float]]:
        """(subgoal_node, goal_node, match_score) or None if not plannable."""
        goal_node, score = self.match_goal(goal_latent)
        if goal_node is None or score < self.tau_goal or self._last_map is None:
            return None
        next_hop = self.bfs_next_hop(self._last_map, goal_node, lookahead=lookahead)
        if next_hop is None:
            return None
        return next_hop, goal_node, score

    def plan_node_path(self, goal_latent: torch.Tensor, avoid_edges=None,
                       allowed_goal_nodes=None) -> Optional[tuple[list[int], int, float]]:
        """Directed node path MAP->goal-matching node for Stage-4b traversal.

        Considers SEVERAL goal-node candidates (the same place is stored under
        multiple view-nodes from different tour passes) and prefers one with a
        DIRECTED path — forward traversal is where alignment and localization
        work; an all-reversed undirected path walks the tour backward facing
        away from every stored keyframe (measured failure). Undirected fallback
        only if no candidate is directed-reachable. ``allowed_goal_nodes`` can
        constrain goal matching with an external deployment-valid discriminator
        such as a raw-image colour gate.
        """
        if not self._keyframes or self._last_map is None:
            return None
        allowed = ({int(node_id) for node_id in allowed_goal_nodes}
                   if allowed_goal_nodes is not None else None)
        node_ids = sorted(self._keyframes)
        keyframes = torch.stack([self._keyframes[i] for i in node_ids])
        scores = keyframes @ F.normalize(goal_latent, dim=-1)
        order = torch.argsort(scores, descending=True)
        candidate_order = order if allowed is not None else order[:8]
        candidates = [(node_ids[int(i)], float(scores[int(i)])) for i in candidate_order
                      if float(scores[int(i)]) >= self.tau_goal
                      and (allowed is None or node_ids[int(i)] in allowed)]
        if not candidates:
            return None
        best = None
        for goal_node, score in candidates:
            path = self._weighted_path(self._last_map, goal_node, reversed_cost=3.0,
                                       avoid_edges=avoid_edges)
            if path is not None and len(path) >= 2:
                n_reversed = sum(1 for i in range(1, len(path))
                                 if (path[i - 1], path[i]) not in self.memory.edges)
                if best is None or n_reversed < best[0]:
                    best = (n_reversed, path, goal_node, score)
                if n_reversed == 0:
                    break
        if best is None:
            return None
        _, path, goal_node, score = best
        return path, goal_node, score

    def _weighted_path(self, start: int, target: int, *, reversed_cost: float = 3.0,
                       avoid_edges=None) -> Optional[list[int]]:
        """Dijkstra over memory edges; traversing an edge against its recorded
        direction is allowed but penalized (a fresh node has only incoming
        edges, so pure-directed search strands; pure-undirected walks the tour
        backward facing away from every keyframe — both measured failures).

        ``avoid_edges``: node pairs that failed traversal (the controller's
        blocked-edge signal); heavily penalized in BOTH directions, not
        removed — if no alternative exists the old route is still returned
        rather than stranding. Without this, replanning after a dead edge is
        idempotent: the unchanged posterior reproduces the identical path and
        the seek re-fails the same edge until the budget dies (v22)."""
        import heapq
        avoid = {(int(a), int(b)) for a, b in (avoid_edges or ())}
        avoid |= {(b, a) for a, b in avoid}
        terminal_nodes = {
            int(node.node_id) for node in getattr(self.memory, "nodes", ())
            if not node.in_filter
        }
        adjacency: dict[int, list[tuple[int, float]]] = {}
        for (a, b) in self.memory.edges:
            dead = 25.0 if (a, b) in avoid else 0.0
            adjacency.setdefault(a, []).append((b, 1.0 + dead))
            adjacency.setdefault(b, []).append((a, reversed_cost + dead))
        if start == target:
            return [start]
        heap, parent, done = [(0.0, start)], {start: None}, set()
        costs = {start: 0.0}
        while heap:
            cost, node = heapq.heappop(heap)
            if node in done:
                continue
            done.add(node)
            if node == target:
                break
            for neighbor, weight in adjacency.get(node, ()):
                # Terminal spurs are goal-only views, not physical shortcuts
                # through the graph. They may be entered only as the target.
                if neighbor in terminal_nodes and neighbor != target:
                    continue
                new_cost = cost + weight
                if new_cost < costs.get(neighbor, float("inf")):
                    costs[neighbor] = new_cost
                    parent[neighbor] = node
                    heapq.heappush(heap, (new_cost, neighbor))
        if target not in parent:
            return None
        path = [target]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        return path[::-1]
