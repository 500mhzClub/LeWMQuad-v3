"""Canonical scene manifest model and deterministic smoke-scene generator."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GraphNode:
    node_id: int
    center_xy_m: tuple[float, float]
    width_m: float
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphEdge:
    source: int
    target: int
    width_m: float
    traversable: bool = True


@dataclass(frozen=True)
class BoxObject:
    object_id: str
    kind: str
    center_xyz_m: tuple[float, float, float]
    size_xyz_m: tuple[float, float, float]
    yaw_rad: float
    material_id: str


@dataclass(frozen=True)
class SpawnSpec:
    xyz_m: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class CameraValidityConstraints:
    min_wall_thickness_m: float
    near_m: float
    far_m: float
    min_camera_clearance_m: float


@dataclass(frozen=True)
class SceneManifest:
    scene_id: str
    family: str
    difficulty_tier: str
    topology_seed: int
    visual_seed: int
    physics_seed: int
    world_bounds_xy_m: tuple[tuple[float, float], tuple[float, float]]
    spawn: SpawnSpec
    graph_nodes: tuple[GraphNode, ...]
    graph_edges: tuple[GraphEdge, ...]
    obstacles: tuple[BoxObject, ...]
    landmarks: tuple[BoxObject, ...]
    camera_constraints: CameraValidityConstraints

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_scene_manifest(scene_seed: int, family: str = "open_obstacle_field") -> SceneManifest:
    """Return a deterministic canonical smoke scene.

    This generator is deliberately small. It provides the contract shape and
    deterministic topology/export path needed before larger scene-family
    generators are added.
    """

    normalized_family = family.strip() or "open_obstacle_field"
    rng = random.Random(f"{normalized_family}:{scene_seed}")
    world_half_extent = 5.0
    grid_side = 3 if normalized_family != "medium_maze" else 4
    spacing = 2.4 if grid_side == 3 else 1.8

    nodes: list[GraphNode] = []
    node_id = 0
    offset = (grid_side - 1) * spacing * 0.5
    for row in range(grid_side):
        for col in range(grid_side):
            x = round(col * spacing - offset, 3)
            y = round(row * spacing - offset, 3)
            tags = ("spawn",) if row == 0 and col == 0 else ()
            nodes.append(GraphNode(node_id=node_id, center_xy_m=(x, y), width_m=1.2, tags=tags))
            node_id += 1

    edges: list[GraphEdge] = []
    for row in range(grid_side):
        for col in range(grid_side):
            node = row * grid_side + col
            if col + 1 < grid_side:
                edges.append(GraphEdge(source=node, target=node + 1, width_m=1.0))
            if row + 1 < grid_side:
                edges.append(GraphEdge(source=node, target=node + grid_side, width_m=1.0))

    obstacle_count = 5 if normalized_family == "open_obstacle_field" else 8
    obstacles: list[BoxObject] = []
    for index in range(obstacle_count):
        size_x = round(rng.uniform(0.35, 0.9), 3)
        size_y = round(rng.uniform(0.35, 1.1), 3)
        size_z = round(rng.uniform(0.25, 0.9), 3)
        x = round(rng.uniform(-3.6, 3.6), 3)
        y = round(rng.uniform(-3.6, 3.6), 3)
        if abs(x) < 0.8 and abs(y) < 0.8:
            x += 1.2
        obstacles.append(
            BoxObject(
                object_id=f"obstacle_{index:03d}",
                kind="box_obstacle",
                center_xyz_m=(x, y, size_z * 0.5),
                size_xyz_m=(size_x, size_y, size_z),
                yaw_rad=round(rng.uniform(-3.14159, 3.14159), 4),
                material_id=f"mat_obstacle_{index % 3}",
            )
        )

    landmarks = (
        BoxObject(
            object_id="landmark_red",
            kind="landmark",
            center_xyz_m=(-4.4, 4.4, 0.75),
            size_xyz_m=(0.35, 0.35, 1.5),
            yaw_rad=0.0,
            material_id="landmark_red",
        ),
        BoxObject(
            object_id="landmark_blue",
            kind="landmark",
            center_xyz_m=(4.4, -4.4, 0.75),
            size_xyz_m=(0.35, 0.35, 1.5),
            yaw_rad=0.0,
            material_id="landmark_blue",
        ),
    )

    spawn = SpawnSpec(xyz_m=(0.0, 0.0, 0.375), quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    constraints = CameraValidityConstraints(
        min_wall_thickness_m=0.08,
        near_m=0.05,
        far_m=200.0,
        min_camera_clearance_m=0.10,
    )

    return SceneManifest(
        scene_id=_stable_scene_id(normalized_family, scene_seed),
        family=normalized_family,
        difficulty_tier="smoke",
        topology_seed=int(scene_seed),
        visual_seed=int(scene_seed) + 100_000,
        physics_seed=int(scene_seed) + 200_000,
        world_bounds_xy_m=((-world_half_extent, -world_half_extent), (world_half_extent, world_half_extent)),
        spawn=spawn,
        graph_nodes=tuple(nodes),
        graph_edges=tuple(edges),
        obstacles=tuple(obstacles),
        landmarks=landmarks,
        camera_constraints=constraints,
    )


def manifest_sha256(manifest: SceneManifest) -> str:
    payload = json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_scene_id(family: str, scene_seed: int) -> str:
    digest = hashlib.sha256(f"{family}:{scene_seed}".encode("utf-8")).hexdigest()
    return f"{family}_{digest[:12]}"
