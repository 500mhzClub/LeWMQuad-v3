"""Online topological memory + top-k Bayes filter (v3 spec §5.4, minimal).

Probe #3 of the Stage 3a reassessment
(``docs/lewm_topological_nav_stage3a_loop_closure_2026-06-09.md``): pairwise
loop-closure verification converged at R≈0.27 @P90 on frozen seq4 — too weak
for single-pair §5.4 decisions — but the deployed mechanism was never a single
pair: the filter aggregates per-step likelihoods over consecutive steps under a
transition prior that shrinks the candidate set to graph neighbors. This module
implements that mechanism so the question can be answered at the level that
matters (§5.5 trajectory coherence), by offline replay over held-out rollouts.

Design notes (the (cell × yaw-bin) decision):
  - Nodes are **view keyframes** — a place *as seen from a heading*. No yaw
    label is needed at inference: the place code is view-selective (the Stage 3a
    finding), so committed nodes are heading-specific by construction, and
    cross-yaw same-place association is graph structure (pivot edges), not
    visual verification.
  - Node embedding = running mean of member belief embeddings, L2-renormalized
    (the pair scorer was trained on normalized embeddings).
  - Per-step update: predict (self-stay + empirical edge transitions), weight
    by calibrated pair-scorer likelihood, top-k truncate (§5.4). Global novelty
    check over ALL nodes (not the top-k); ``new_node_streak`` consecutive
    sub-``tau_new`` steps commit a new node with an edge from the previous MAP.

Genesis-free; the scorer is any callable ``(query (D,), nodes (N, D)) -> (N,)``
probabilities, so unit tests run on synthetic embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class MemoryNode:
    node_id: int
    embedding: torch.Tensor          # (D,) L2-normalized running mean
    n_members: int = 1
    member_labels: list = field(default_factory=list)  # eval-only bookkeeping


class OnlineTopologicalMemory:
    def __init__(
        self,
        scorer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        tau_new: float,
        new_node_streak: int = 3,
        top_k: int = 8,
        self_stay_prob: float = 0.6,
        uniform_leak: float = 0.05,
        likelihood_floor: float = 1e-4,
        update_node_embedding: bool = True,
    ) -> None:
        self.scorer = scorer
        self.tau_new = float(tau_new)
        self.new_node_streak = int(new_node_streak)
        self.top_k = int(top_k)
        self.self_stay_prob = float(self_stay_prob)
        self.uniform_leak = float(uniform_leak)
        self.likelihood_floor = float(likelihood_floor)
        self.update_node_embedding = bool(update_node_embedding)

        self.nodes: list[MemoryNode] = []
        self.edges: dict[tuple[int, int], int] = {}
        self.posterior: dict[int, float] = {}
        self._novelty_streak = 0
        self._novelty_buffer: list[torch.Tensor] = []
        self._previous_map: int | None = None

    # ------------------------------------------------------------------ #

    def _node_matrix(self) -> torch.Tensor:
        return torch.stack([n.embedding for n in self.nodes])

    def _commit_node(self, label=None) -> int:
        embedding = F.normalize(torch.stack(self._novelty_buffer).mean(0), dim=-1)
        node = MemoryNode(node_id=len(self.nodes), embedding=embedding)
        if label is not None:
            node.member_labels.append(label)
        self.nodes.append(node)
        if self._previous_map is not None:
            key = (self._previous_map, node.node_id)
            self.edges[key] = self.edges.get(key, 0) + 1
        self.posterior = {node.node_id: 1.0}
        self._previous_map = node.node_id
        self._novelty_streak = 0
        self._novelty_buffer = []
        return node.node_id

    def _predict(self) -> dict[int, float]:
        """Transition prior + a small uniform leak over ALL nodes.

        Without the leak, a never-traversed transition has zero prior mass and
        the posterior cannot jump to a known node over a new edge — the MAP
        sticks on the previous node and pollutes it with wrong-place frames
        (caught by the unit test). The leak is the standard discrete-filter
        remedy; novelty detection is unaffected (it checks raw likelihoods).
        """
        predicted: dict[int, float] = {}
        outgoing: dict[int, list[tuple[int, int]]] = {}
        for (src, dst), count in self.edges.items():
            outgoing.setdefault(src, []).append((dst, count))
        keep = 1.0 - self.uniform_leak
        for node_id, weight in self.posterior.items():
            neighbors = outgoing.get(node_id, [])
            total = sum(c for _, c in neighbors)
            stay = self.self_stay_prob if total else 1.0
            predicted[node_id] = predicted.get(node_id, 0.0) + keep * weight * stay
            for dst, count in neighbors:
                move = keep * weight * (1.0 - self.self_stay_prob) * count / total
                predicted[dst] = predicted.get(dst, 0.0) + move
        leak_each = self.uniform_leak / len(self.nodes)
        for node in self.nodes:
            predicted[node.node_id] = predicted.get(node.node_id, 0.0) + leak_each
        return predicted

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def update(self, embedding: torch.Tensor, label=None) -> int | None:
        """One filter step. Returns the MAP node id (None during cold start)."""
        if self.nodes:
            likelihoods = self.scorer(embedding, self._node_matrix()).clamp(
                self.likelihood_floor, 1.0 - self.likelihood_floor
            )
            global_max = float(likelihoods.max())
        else:
            likelihoods, global_max = None, 0.0

        # Global novelty check (over ALL nodes, not the top-k).
        if global_max < self.tau_new:
            self._novelty_streak += 1
            self._novelty_buffer.append(embedding)
            if self._novelty_streak >= self.new_node_streak:
                return self._commit_node(label)
        else:
            self._novelty_streak = 0
            self._novelty_buffer = []

        if not self.nodes:
            return None

        predicted = self._predict() if self.posterior else {
            n.node_id: 1.0 / len(self.nodes) for n in self.nodes
        }
        updated = {nid: w * float(likelihoods[nid]) for nid, w in predicted.items() if w > 0}
        if not updated or sum(updated.values()) <= 0:
            updated = {nid: float(likelihoods[nid]) for nid in range(len(self.nodes))}
        ranked = sorted(updated.items(), key=lambda kv: kv[1], reverse=True)[: self.top_k]
        total = sum(w for _, w in ranked)
        self.posterior = {nid: w / total for nid, w in ranked}

        map_node = max(self.posterior.items(), key=lambda kv: kv[1])[0]
        # Edge counting between consecutive MAP nodes (posterior-weighted
        # contribution simplified to the MAP transition; adequate for replay).
        if self._previous_map is not None and self._previous_map != map_node:
            key = (self._previous_map, map_node)
            self.edges[key] = self.edges.get(key, 0) + 1
        self._previous_map = map_node

        node = self.nodes[map_node]
        if label is not None:
            node.member_labels.append(label)
        if self.update_node_embedding and global_max >= self.tau_new:
            mean = node.embedding * node.n_members + embedding
            node.n_members += 1
            node.embedding = F.normalize(mean / node.n_members, dim=-1)
        return map_node

    # ------------------------------------------------------------------ #
    # Evaluation helpers (§5.5 / §6.1 node-purity rule)
    # ------------------------------------------------------------------ #

    def node_majority_labels(self, tau_purity: float = 0.8) -> dict[int, tuple]:
        """node_id -> (majority_label, purity, reliable)."""
        out = {}
        for node in self.nodes:
            if not node.member_labels:
                continue
            counts: dict = {}
            for label in node.member_labels:
                counts[label] = counts.get(label, 0) + 1
            majority = max(counts.items(), key=lambda kv: kv[1])
            purity = majority[1] / len(node.member_labels)
            out[node.node_id] = (majority[0], purity, purity >= tau_purity)
        return out
