"""Genesis scene-spec exporter for canonical LeWM scene manifests."""

from __future__ import annotations

from typing import Any

from lewm_worlds.manifest import SceneManifest, manifest_sha256


def export_genesis_scene(manifest: SceneManifest) -> dict[str, Any]:
    """Return a dependency-free Genesis construction spec.

    The spec is consumed by `lewm_genesis.scene_builder`. Keeping this exporter
    free of a hard Genesis import lets CI validate manifest parity before the
    renderer is installed.
    """

    return {
        "scene_id": manifest.scene_id,
        "family": manifest.family,
        "manifest_sha256": manifest_sha256(manifest),
        "physics_seed": manifest.physics_seed,
        "world_bounds_xy_m": manifest.world_bounds_xy_m,
        "spawn": {
            "xyz_m": manifest.spawn.xyz_m,
            "quat_wxyz": manifest.spawn.quat_wxyz,
        },
        "objects": [
            {
                "object_id": obj.object_id,
                "kind": obj.kind,
                "center_xyz_m": obj.center_xyz_m,
                "size_xyz_m": obj.size_xyz_m,
                "yaw_rad": obj.yaw_rad,
                "material_id": obj.material_id,
            }
            for obj in manifest.static_objects
        ],
        "graph": {
            "nodes": [node.__dict__ for node in manifest.graph_nodes],
            "edges": [edge.__dict__ for edge in manifest.graph_edges],
        },
        "camera_constraints": manifest.camera_constraints.__dict__,
    }
