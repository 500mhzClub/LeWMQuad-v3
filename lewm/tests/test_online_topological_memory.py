"""Unit tests for lewm.memory.online_topological_memory (probe #3 mechanics)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from lewm.memory.online_topological_memory import MemoryNode, OnlineTopologicalMemory  # noqa: E402
from lewm.memory.topological_navigator import TopologicalNavigator  # noqa: E402


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


def test_frozen_memory_localizes_without_mutating_map() -> None:
    centers = _make_world(n_places=4)
    embeddings, labels = _trajectory(centers, [0, 1, 2], dwell=4, noise=0.02)
    memory = OnlineTopologicalMemory(_cosine_prob_scorer, tau_new=0.5, new_node_streak=2, top_k=5)
    for embedding, label in zip(embeddings, labels):
        memory.update(embedding, label=label)

    memory.frozen = True
    n_nodes = len(memory.nodes)
    edges = dict(memory.edges)
    node_state = [(n.n_members, list(n.member_labels), n.embedding.clone()) for n in memory.nodes]
    for _ in range(5):
        memory.update(centers[3], label=3)

    assert len(memory.nodes) == n_nodes
    assert memory.edges == edges
    assert memory._novelty_streak == 0 and not memory._novelty_buffer
    for node, (n_members, member_labels, embedding) in zip(memory.nodes, node_state):
        assert node.n_members == n_members
        assert node.member_labels == member_labels
        assert torch.equal(node.embedding, embedding)
    print("PASS frozen memory localizes without map mutation")


def test_terminal_spur_is_excluded_from_filter() -> None:
    centers = _make_world(n_places=2)
    memory = OnlineTopologicalMemory(_cosine_prob_scorer, tau_new=0.5, new_node_streak=2, top_k=5)
    for embedding, label in zip(*_trajectory(centers, [0], dwell=4, noise=0.02)):
        memory.update(embedding, label=label)
    spur_id = len(memory.nodes)
    memory.nodes.append(MemoryNode(node_id=spur_id, embedding=centers[1], in_filter=False))
    memory.edges[(0, spur_id)] = 1
    memory.frozen = True

    for _ in range(4):
        assignment = memory.update(centers[1])
        assert assignment != spur_id
        assert spur_id not in memory.posterior
    print("PASS terminal spur excluded from localization filter")


def test_terminal_spur_records_forward_bearing() -> None:
    class MeanEncoder:
        def __call__(self, window):
            return F.normalize(window.mean(dim=1), dim=-1)

    navigator = TopologicalNavigator(
        MeanEncoder(), _cosine_prob_scorer, history=4, tau_new=0.5,
    )
    anchor = F.normalize(torch.tensor([1.0, 0.0]), dim=-1)
    navigator.memory.nodes.append(MemoryNode(node_id=0, embedding=anchor))
    spur_id = navigator.insert_spur(
        torch.tensor([0.0, 1.0]), "goal-frame", 0, label=1, bearing_rad=1.25,
    )

    assert navigator._edge_bearings == {(0, spur_id): 1.25}
    assert (0, spur_id) in navigator.memory.edges
    assert (spur_id, 0) in navigator.memory.edges
    assert not navigator.memory.nodes[spur_id].in_filter
    print("PASS terminal spur records forward traversal bearing")


def test_weighted_path_penalizes_failed_edge_both_directions() -> None:
    navigator = TopologicalNavigator.__new__(TopologicalNavigator)
    navigator.memory = SimpleNamespace(edges={
        (0, 1): 1, (1, 3): 1,
        (0, 2): 1, (2, 3): 1,
    })
    assert navigator._weighted_path(0, 3) == [0, 1, 3]
    assert navigator._weighted_path(0, 3, avoid_edges={(0, 1)}) == [0, 2, 3]
    assert navigator._weighted_path(1, 0, avoid_edges={(0, 1)}) == [1, 3, 2, 0]
    print("PASS failed edge penalized in both traversal directions")


def test_weighted_path_does_not_transit_terminal_spur() -> None:
    navigator = TopologicalNavigator.__new__(TopologicalNavigator)
    navigator.memory = SimpleNamespace(
        nodes=[
            MemoryNode(node_id=0, embedding=torch.tensor([1.0])),
            MemoryNode(node_id=1, embedding=torch.tensor([1.0]), in_filter=False),
            MemoryNode(node_id=2, embedding=torch.tensor([1.0])),
            MemoryNode(node_id=3, embedding=torch.tensor([1.0])),
        ],
        edges={(0, 1): 1, (1, 3): 1, (0, 2): 1, (2, 3): 1},
    )

    assert navigator._weighted_path(0, 1) == [0, 1]
    assert navigator._weighted_path(0, 3) == [0, 2, 3]
    print("PASS terminal spur is target-only in weighted routing")


def test_plan_node_path_respects_allowed_goal_nodes() -> None:
    navigator = TopologicalNavigator.__new__(TopologicalNavigator)
    navigator.memory = SimpleNamespace(edges={(0, 1): 1, (0, 2): 1})
    navigator._last_map = 0
    navigator.tau_goal = 0.8
    navigator._keyframes = {
        1: torch.tensor([1.0, 0.0]),
        2: F.normalize(torch.tensor([0.99, 0.1]), dim=-1),
    }
    goal = torch.tensor([1.0, 0.0])

    path, goal_node, _score = navigator.plan_node_path(goal)
    assert path == [0, 1] and goal_node == 1
    path, goal_node, _score = navigator.plan_node_path(goal, allowed_goal_nodes={2})
    assert path == [0, 2] and goal_node == 2
    assert navigator.plan_node_path(goal, allowed_goal_nodes=set()) is None
    print("PASS goal-node allowlist constrains replanning")


if __name__ == "__main__":
    test_commits_nodes_and_localizes_coherently()
    test_revisit_closes_loop_not_duplicate_explosion()
    test_cold_start_and_novelty_streak()
    test_frozen_memory_localizes_without_mutating_map()
    test_terminal_spur_is_excluded_from_filter()
    test_terminal_spur_records_forward_bearing()
    test_weighted_path_penalizes_failed_edge_both_directions()
    test_weighted_path_does_not_transit_terminal_spur()
    test_plan_node_path_respects_allowed_goal_nodes()
    print("ALL PASS")
