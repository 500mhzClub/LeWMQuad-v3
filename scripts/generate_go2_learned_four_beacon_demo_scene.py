#!/usr/bin/env python3
"""Generate a minimal four-beacon scene for the fully learned Go2 demo gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lewm_worlds"))

from lewm_worlds.exporters.to_gazebo_sdf import export_gazebo_sdf  # noqa: E402
from lewm_worlds.exporters.to_genesis import export_genesis_scene  # noqa: E402
from lewm_worlds.labels.topology import topology_summary  # noqa: E402
from lewm_worlds.manifest import (  # noqa: E402
    BoxObject,
    CameraExtrinsicJitter,
    CameraValidityConstraints,
    GraphEdge,
    GraphNode,
    LightingSpec,
    MaterialOverride,
    PhysicsRandomization,
    SceneManifest,
    SpawnSpec,
    VisualRandomization,
    manifest_sha256,
)


DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / ".generated" / "scene_corpus" / "learned_four_beacon_demo_20260703"
)
DEFAULT_SCENE_ID = "learned_four_beacon_open_0001"
DEFAULT_FAMILY = "open_obstacle_field"


def build_manifest(*, scene_id: str = DEFAULT_SCENE_ID, family: str = DEFAULT_FAMILY) -> SceneManifest:
    landmarks: list[BoxObject] = []
    for index, (color, x_m, y_m) in enumerate(
        (
            ("red", 0.90, -0.27),
            ("yellow", 0.95, -0.09),
            ("blue", 1.00, 0.09),
            ("green", 1.05, 0.27),
        )
    ):
        landmarks.append(
            BoxObject(
                object_id=f"landmark_{index:02d}_landmark_{color}",
                kind="landmark",
                center_xyz_m=(x_m, y_m, 0.44),
                size_xyz_m=(0.16, 0.16, 0.88),
                yaw_rad=0.0,
                material_id=f"landmark_{color}",
            )
        )

    graph_nodes = (
        GraphNode(node_id=0, center_xy_m=(0.0, 0.0), width_m=1.2, tags=("spawn",)),
        GraphNode(node_id=1, center_xy_m=(0.90, -0.27), width_m=1.0, tags=("landmark_red",)),
        GraphNode(node_id=2, center_xy_m=(0.95, -0.09), width_m=1.0, tags=("landmark_yellow",)),
        GraphNode(node_id=3, center_xy_m=(1.00, 0.09), width_m=1.0, tags=("landmark_blue",)),
        GraphNode(node_id=4, center_xy_m=(1.05, 0.27), width_m=1.0, tags=("landmark_green",)),
    )
    graph_edges = tuple(
        GraphEdge(source=0, target=node_id, width_m=1.0, traversable=True)
        for node_id in range(1, 5)
    )

    return SceneManifest(
        scene_id=scene_id,
        family=family,
        difficulty_tier="demo",
        topology_seed=2026070301,
        visual_seed=2026070302,
        physics_seed=2026070303,
        world_bounds_xy_m=((-2.0, -2.0), (2.4, 2.0)),
        spawn=SpawnSpec(xyz_m=(0.0, 0.0, 0.375), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        obstacles=(),
        landmarks=tuple(landmarks),
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.12,
            near_m=0.08,
            far_m=200.0,
            min_camera_clearance_m=0.1,
        ),
        split="train",
        walls=(),
        visual_randomization=VisualRandomization(
            material_overrides=(
                MaterialOverride(material_id="floor", rgba=(0.42, 0.44, 0.42, 1.0)),
                MaterialOverride(material_id="obstacle", rgba=(0.45, 0.45, 0.48, 1.0)),
                MaterialOverride(material_id="wall", rgba=(0.52, 0.52, 0.54, 1.0)),
            ),
            lighting=LightingSpec(
                direction=(0.2, 0.1, -0.97),
                diffuse_rgb=(0.85, 0.85, 0.82),
                specular_rgb=(0.2, 0.2, 0.2),
                ambient_rgb=(0.24, 0.24, 0.24),
            ),
            distractor_objects=(),
        ),
        physics_randomization=PhysicsRandomization(
            floor_friction_mu=0.9,
            floor_restitution=0.01,
            obstacle_friction_mu=0.85,
            obstacle_restitution=0.01,
        ),
        camera_extrinsic_jitter=CameraExtrinsicJitter(
            xyz_offset_m=(0.0, 0.0, 0.0),
            rpy_offset_rad=(0.0, 0.0, 0.0),
        ),
    )


def write_scene(output_root: Path, manifest: SceneManifest) -> Path:
    scene_dir = output_root / "train" / manifest.family / manifest.scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    export_gazebo_sdf(manifest, scene_dir)
    (scene_dir / "genesis_scene.json").write_text(
        json.dumps(export_genesis_scene(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (scene_dir / "topology.json").write_text(
        json.dumps(topology_summary(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return scene_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--family", default=DEFAULT_FAMILY)
    args = parser.parse_args()

    manifest = build_manifest(scene_id=str(args.scene_id), family=str(args.family))
    scene_dir = write_scene(args.output_root, manifest)
    print(f"scene_dir={scene_dir}")
    print(f"manifest_sha256={manifest_sha256(manifest)}")
    print(f"scene_id={manifest.scene_id}")
    print(f"family={manifest.family}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
