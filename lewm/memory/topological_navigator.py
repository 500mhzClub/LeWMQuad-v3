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

from lewm.memory.online_topological_memory import OnlineTopologicalMemory
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

    def bfs_path(self, start: int, target: int) -> Optional[list[int]]:
        """Shortest undirected node path start->target incl. both ends (None if disconnected)."""
        if start == target:
            return [start]
        adjacency: dict[int, set[int]] = {}
        for (a, b) in self.memory.edges:
            adjacency.setdefault(a, set()).add(b)
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
