"""Read-only graph accessor over a :class:`SceneManifest`.

This module is the privileged-side scene oracle. It is used at data
generation time by collector policies (route teacher, frontier teacher,
recovery curriculum) and at label-derivation time by Phase A1 metadata
recovery. **It is never an input to the deployed model** — see the
privileged-leak rule in ``docs/v3_hjepa_plan.md`` §3.4 and
``docs/fresh_retrain_data_spec.md`` §2.

All graph indices are scene-scoped: the caller is responsible for pairing
``cell_id`` with the corresponding ``scene_id``.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from lewm_worlds.manifest import BoxObject, SceneManifest


@dataclass(frozen=True)
class CellHit:
    """Result of locating a world-frame xy in the scene's cell graph."""

    cell_id: int
    distance_m: float


class SceneGraph:
    """Pre-computed adjacency, BFS, and geometry queries for a scene.

    The class is intended to be constructed once per scene (during scene
    build) and reused for every per-block decision. All methods are pure
    and depend only on the manifest, so a collector can call them safely
    across parallel envs sharing the same scene.
    """

    def __init__(self, manifest: SceneManifest) -> None:
        self.manifest = manifest
        self._node_xy = np.asarray(
            [node.center_xy_m for node in manifest.graph_nodes], dtype=np.float32
        )
        self._node_tags = tuple(node.tags for node in manifest.graph_nodes)
        self._adjacency: dict[int, list[int]] = {
            node.node_id: [] for node in manifest.graph_nodes
        }
        for edge in manifest.graph_edges:
            if not edge.traversable:
                continue
            self._adjacency[edge.source].append(edge.target)
            self._adjacency[edge.target].append(edge.source)

        # Pre-compute a flat array of wall + obstacle AABBs for fast clearance
        # queries. Each entry is (cx, cy, half_x, half_y, yaw). Landmarks are
        # not treated as walls — they're navigation targets, not occluders.
        boxes: list[tuple[float, float, float, float, float]] = []
        for obj in (*manifest.walls, *manifest.obstacles):
            cx, cy, _cz = obj.center_xyz_m
            sx, sy, _sz = obj.size_xyz_m
            boxes.append((float(cx), float(cy), float(sx) * 0.5, float(sy) * 0.5, float(obj.yaw_rad)))
        self._box_aabbs = np.asarray(
            boxes if boxes else np.zeros((0, 5)),
            dtype=np.float32,
        ).reshape(-1, 5)

        # Landmark cells: nearest graph node per landmark, used by the
        # route teacher for goal selection.
        self._landmark_cells: tuple[tuple[str, int], ...] = tuple(
            (lm.object_id, self._nearest_node_index((lm.center_xyz_m[0], lm.center_xyz_m[1])))
            for lm in manifest.landmarks
        )

    # ------------------------------------------------------------------
    # Topology queries
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return int(self._node_xy.shape[0])

    @property
    def node_xy(self) -> np.ndarray:
        return self._node_xy

    @property
    def landmark_cells(self) -> tuple[tuple[str, int], ...]:
        """Tuple of ``(object_id, cell_id)`` for every landmark in the scene."""

        return self._landmark_cells

    def neighbors(self, cell_id: int) -> tuple[int, ...]:
        return tuple(self._adjacency.get(int(cell_id), ()))

    def bfs_distance(self, start: int, goal: int) -> int | None:
        """Return graph distance (in hops) on the traversable subgraph, or None."""

        if start == goal:
            return 0
        visited = {int(start)}
        frontier: deque[tuple[int, int]] = deque([(int(start), 0)])
        while frontier:
            node, depth = frontier.popleft()
            for neighbour in self._adjacency.get(node, ()):
                if neighbour in visited:
                    continue
                if neighbour == goal:
                    return depth + 1
                visited.add(neighbour)
                frontier.append((neighbour, depth + 1))
        return None

    def next_waypoint(self, start: int, goal: int) -> int | None:
        """Return the next neighbour of ``start`` that lies on a shortest path."""

        if start == goal:
            return None
        parent: dict[int, int] = {int(start): -1}
        frontier: deque[int] = deque([int(start)])
        found = False
        while frontier and not found:
            node = frontier.popleft()
            for neighbour in self._adjacency.get(node, ()):
                if neighbour in parent:
                    continue
                parent[neighbour] = node
                if neighbour == goal:
                    found = True
                    break
                frontier.append(neighbour)
        if int(goal) not in parent:
            return None
        cursor = int(goal)
        while parent[cursor] != int(start):
            cursor = parent[cursor]
            if cursor == -1:
                return None
        return cursor

    def reachable_cells(self, start: int) -> set[int]:
        """Return the set of node ids reachable from ``start`` on traversable edges."""

        visited: set[int] = {int(start)}
        frontier: deque[int] = deque([int(start)])
        while frontier:
            node = frontier.popleft()
            for neighbour in self._adjacency.get(node, ()):
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                frontier.append(neighbour)
        return visited

    def dead_end_cells(self) -> tuple[int, ...]:
        """Return cell ids whose traversable degree is ≤ 1 (dead ends)."""

        return tuple(
            sorted(
                node.node_id
                for node in self.manifest.graph_nodes
                if len(self._adjacency.get(node.node_id, ())) <= 1
            )
        )

    def spawn_cells(self) -> tuple[int, ...]:
        """Return cell ids tagged as canonical spawn cells in the manifest."""

        return tuple(
            node.node_id
            for node in self.manifest.graph_nodes
            if "spawn" in node.tags
        )

    # ------------------------------------------------------------------
    # World-frame queries
    # ------------------------------------------------------------------

    def locate(self, xy_world: tuple[float, float]) -> CellHit:
        """Return the nearest cell to ``xy_world`` and the Euclidean distance."""

        idx = self._nearest_node_index(xy_world)
        cx, cy = float(self._node_xy[idx, 0]), float(self._node_xy[idx, 1])
        dist = math.hypot(xy_world[0] - cx, xy_world[1] - cy)
        return CellHit(cell_id=idx, distance_m=dist)

    def cell_center(self, cell_id: int) -> tuple[float, float]:
        idx = int(cell_id)
        return float(self._node_xy[idx, 0]), float(self._node_xy[idx, 1])

    def clearance_to_walls(self, xy_world: tuple[float, float]) -> float:
        """Return distance from ``xy_world`` to the nearest wall/obstacle AABB.

        AABBs are evaluated in world frame after rotating the query point by
        ``-yaw`` per box (so we can test against an axis-aligned half-extent).
        Returns ``+inf`` if the scene has no walls or obstacles.
        """

        if self._box_aabbs.shape[0] == 0:
            return float("inf")
        x, y = float(xy_world[0]), float(xy_world[1])
        cx = self._box_aabbs[:, 0]
        cy = self._box_aabbs[:, 1]
        hx = self._box_aabbs[:, 2]
        hy = self._box_aabbs[:, 3]
        yaw = self._box_aabbs[:, 4]
        cos_y = np.cos(-yaw)
        sin_y = np.sin(-yaw)
        dx_world = x - cx
        dy_world = y - cy
        local_x = cos_y * dx_world - sin_y * dy_world
        local_y = sin_y * dx_world + cos_y * dy_world
        outside_x = np.maximum(0.0, np.abs(local_x) - hx)
        outside_y = np.maximum(0.0, np.abs(local_y) - hy)
        dists = np.hypot(outside_x, outside_y)
        return float(np.min(dists))

    # ------------------------------------------------------------------
    # Spawn-pose sampling
    # ------------------------------------------------------------------

    def sample_spawn_pose(
        self,
        rng: random.Random,
        *,
        clearance_floor_m: float = 0.20,
        max_attempts: int = 50,
        spawn_z_m: float = 0.375,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float], int]:
        """Sample a random spawn pose at a random cell with random yaw.

        Returns ``(xyz_m, quat_wxyz, cell_id)``. Cells whose center clearance
        falls below ``clearance_floor_m`` are rejected; if no cell can be
        sampled within ``max_attempts`` tries, falls back to the manifest's
        original spawn pose.
        """

        candidates = list(range(self.n_nodes))
        if not candidates:
            xyz = self.manifest.spawn.xyz_m
            quat = self.manifest.spawn.quat_wxyz
            return xyz, quat, -1
        rng.shuffle(candidates)
        for cell_id in candidates[:max_attempts]:
            xy = self.cell_center(cell_id)
            if self.clearance_to_walls(xy) < clearance_floor_m:
                continue
            yaw = rng.uniform(-math.pi, math.pi)
            return (
                (float(xy[0]), float(xy[1]), float(spawn_z_m)),
                _yaw_to_wxyz(yaw),
                int(cell_id),
            )
        # Fall back to the manifest spawn so the scene always boots.
        fallback_xy = self.manifest.spawn.xyz_m
        return (
            (float(fallback_xy[0]), float(fallback_xy[1]), float(spawn_z_m)),
            self.manifest.spawn.quat_wxyz,
            int(self.locate((fallback_xy[0], fallback_xy[1])).cell_id),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _nearest_node_index(self, xy_world: tuple[float, float]) -> int:
        if self._node_xy.shape[0] == 0:
            raise ValueError("scene has no graph nodes")
        diffs = self._node_xy - np.asarray(
            [float(xy_world[0]), float(xy_world[1])], dtype=np.float32
        )
        sq = np.einsum("ij,ij->i", diffs, diffs)
        return int(np.argmin(sq))


def _yaw_to_wxyz(yaw_rad: float) -> tuple[float, float, float, float]:
    half = float(yaw_rad) * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def wrap_angle_pi(angle: float) -> float:
    """Wrap an angle in radians into ``[-pi, pi]``."""

    return float(((angle + math.pi) % (2.0 * math.pi)) - math.pi)


def bearing_from_to(
    src_xy: Iterable[float], dst_xy: Iterable[float]
) -> float:
    """Return the world-frame bearing (radians) from ``src_xy`` to ``dst_xy``."""

    sx, sy = (float(v) for v in src_xy)
    dx, dy = (float(v) for v in dst_xy)
    return math.atan2(dy - sy, dx - sx)
