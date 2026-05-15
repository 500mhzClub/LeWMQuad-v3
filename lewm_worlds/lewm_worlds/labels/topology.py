"""Topology labels derived directly from canonical scene manifests."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from lewm_worlds.manifest import SceneManifest


def topology_summary(manifest: SceneManifest) -> dict[str, Any]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for edge in manifest.graph_edges:
        if not edge.traversable:
            continue
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)

    dead_ends = sorted(
        node.node_id for node in manifest.graph_nodes if len(adjacency[node.node_id]) <= 1
    )
    component_count = _component_count([node.node_id for node in manifest.graph_nodes], adjacency)
    edge_count = sum(1 for edge in manifest.graph_edges if edge.traversable)
    node_count = len(manifest.graph_nodes)
    cycle_count = max(0, edge_count - node_count + component_count)

    return {
        "scene_id": manifest.scene_id,
        "node_count": node_count,
        "edge_count": edge_count,
        "component_count": component_count,
        "dead_end_node_ids": dead_ends,
        "cycle_count": cycle_count,
        "landmark_object_ids": [obj.object_id for obj in manifest.landmarks],
    }


def _component_count(node_ids: list[int], adjacency: dict[int, set[int]]) -> int:
    unseen = set(node_ids)
    count = 0
    while unseen:
        count += 1
        start = unseen.pop()
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
    return count
