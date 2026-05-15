"""Gazebo SDF exporter for canonical LeWM scene manifests."""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from lewm_worlds.manifest import BoxObject, SceneManifest, manifest_sha256


def export_gazebo_sdf(manifest: SceneManifest, out_dir: Path) -> Path:
    """Write `world.sdf` and `manifest.json` for a canonical scene."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    world_path = out_dir / "world.sdf"
    manifest_path = out_dir / "manifest.json"

    world_path.write_text(_render_world(manifest), encoding="utf-8")
    manifest_payload = manifest.to_dict()
    manifest_payload["manifest_sha256"] = manifest_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return world_path


def _render_world(manifest: SceneManifest) -> str:
    models = "\n".join(_render_box_model(obj) for obj in (*manifest.obstacles, *manifest.landmarks))
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <!-- scene_id: {escape(manifest.scene_id)} -->
  <world name="default">
    <physics name="1ms" type="bullet-featherstone">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry>
          <material><ambient>0.45 0.47 0.42 1</ambient><diffuse>0.45 0.47 0.42 1</diffuse></material>
        </visual>
      </link>
    </model>
{models}
  </world>
</sdf>
"""


def _render_box_model(obj: BoxObject) -> str:
    x, y, z = obj.center_xyz_m
    sx, sy, sz = obj.size_xyz_m
    color = _material_rgba(obj.material_id)
    return f"""    <model name="{escape(obj.object_id)}">
      <static>true</static>
      <pose>{x} {y} {z} 0 0 {obj.yaw_rad}</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
          <material><ambient>{color}</ambient><diffuse>{color}</diffuse></material>
        </visual>
      </link>
    </model>"""


def _material_rgba(material_id: str) -> str:
    palette = {
        "landmark_red": "0.85 0.12 0.08 1",
        "landmark_blue": "0.10 0.22 0.85 1",
        "mat_obstacle_0": "0.55 0.55 0.52 1",
        "mat_obstacle_1": "0.42 0.48 0.38 1",
        "mat_obstacle_2": "0.56 0.42 0.35 1",
    }
    return palette.get(material_id, "0.5 0.5 0.5 1")
