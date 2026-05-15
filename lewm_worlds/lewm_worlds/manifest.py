"""Canonical scene manifest schema and dispatcher into per-family builders."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
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
    split: str | None = None
    walls: tuple[BoxObject, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def static_objects(self) -> tuple[BoxObject, ...]:
        return (*self.walls, *self.obstacles, *self.landmarks)


def build_scene_manifest(
    scene_seed: int,
    family: str = "open_obstacle_field",
    *,
    split: str | None = None,
    difficulty_tier: str | None = None,
) -> SceneManifest:
    """Return a deterministic scene manifest for the requested family.

    The actual per-family geometry lives in :mod:`lewm_worlds.families`. This
    function is the stable entry point used by the smoke generator, the corpus
    builder, and downstream renderers.
    """

    from lewm_worlds.families import build_family_manifest

    return build_family_manifest(
        scene_seed=int(scene_seed),
        family=family,
        split=split,
        difficulty_tier=difficulty_tier,
    )


def manifest_sha256(manifest: SceneManifest) -> str:
    payload = json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_scene_id(family: str, scene_seed: int) -> str:
    digest = hashlib.sha256(f"{family}:{scene_seed}".encode("utf-8")).hexdigest()
    return f"{family}_{digest[:12]}"
