"""Unit tests for lewm.memory.online_topological_memory (probe #3 mechanics)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from lewm.memory.online_topological_memory import OnlineTopologicalMemory  # noqa: E402


def _cosine_prob_scorer(query: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
    # Sharp sigmoid on cosine — mimics a calibrated pair head (near-1 for same
    # view, near-0 for different), unlike a flat (cos+1)/2 mapping.
    return torch.sigmoid((nodes @ query - 0.9) * 40.0)


def _make_world(n_places: int = 6, dim: int = 32, seed: int = 0):
    torch.manual_seed(seed)
    centers = F.normalize(torch.randn(n_places, dim), dim=-1)
    return centers


def _trajectory(centers, visit_sequence, dwell: int = 6, noise: float = 0.05, seed: int = 1):
    torch.manual_seed(seed)
    embeddings, labels = [], []
    for place in visit_sequence:
        for _ in range(dwell):
            embeddings.append(F.normalize(centers[place] + noise * torch.randn(centers.shape[-1]), dim=-1))
            labels.append(int(place))
    return embeddings, labels


def test_commits_nodes_and_localizes_coherently() -> None:
    centers = _make_world()
    visit = [0, 1, 2, 3, 2, 1, 0, 4, 5, 4, 0, 1, 2]
    embeddings, labels = _trajectory(centers, visit)
    memory = OnlineTopologicalMemory(_cosine_prob_scorer, tau_new=0.5, new_node_streak=2, top_k=5)
    assignments = [memory.update(e, label=l) for e, l in zip(embeddings, labels)]

    majority = memory.node_majority_labels()
    n_places_visited = len(set(labels))
    assert len(memory.nodes) >= n_places_visited, "missed places"
    # False merges: every node must be pure (distinct synthetic places are separable).
    assert all(reliable for _, _, reliable in majority.values()), f"impure nodes: {majority}"

    correct = total = 0
    for assignment, label in zip(assignments, labels):
        if assignment is None or assignment not in majority:
            continue
        total += 1
        correct += int(majority[assignment][0] == label)
    coherence = correct / max(total, 1)
    assert coherence > 0.9, f"coherence too low: {coherence:.3f}"
    print(f"PASS coherent localization (nodes={len(memory.nodes)}, coherence={coherence:.3f})")


def test_revisit_closes_loop_not_duplicate_explosion() -> None:
    centers = _make_world()
    # Visit 3 places, then loop the same circuit 4 more times.
    visit = [0, 1, 2] * 5
    embeddings, labels = _trajectory(centers, visit, dwell=5, noise=0.04)
    memory = OnlineTopologicalMemory(_cosine_prob_scorer, tau_new=0.5, new_node_streak=2, top_k=5)
    for e, l in zip(embeddings, labels):
        memory.update(e, label=l)
    # Fragmentation allowed but bounded: revisits should mostly re-localize.
    assert len(memory.nodes) <= 6, f"duplicate explosion: {len(memory.nodes)} nodes for 3 places"
    assert len(memory.edges) >= 2, "no edges learned"
    print(f"PASS loop revisits reuse nodes (nodes={len(memory.nodes)}, edges={len(memory.edges)})")


def test_cold_start_and_novelty_streak() -> None:
    centers = _make_world()
    embeddings, labels = _trajectory(centers, [0], dwell=4, noise=0.02)
    memory = OnlineTopologicalMemory(_cosine_prob_scorer, tau_new=0.5, new_node_streak=3, top_k=5)
    out = [memory.update(e, label=l) for e, l in zip(embeddings, labels)]
    assert out[0] is None and out[1] is None, "committed before streak satisfied"
    assert out[2] == 0, "first node not committed at streak"
    assert len(memory.nodes) == 1
    print("PASS cold start + novelty streak")


if __name__ == "__main__":
    test_commits_nodes_and_localizes_coherently()
    test_revisit_closes_loop_not_duplicate_explosion()
    test_cold_start_and_novelty_streak()
    print("ALL PASS")
