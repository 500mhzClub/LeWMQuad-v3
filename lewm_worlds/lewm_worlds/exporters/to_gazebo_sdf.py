"""Gazebo SDF exporter for canonical LeWM scene manifests.

Visual and physics randomization (data spec §14) is applied directly:

- The ground plane uses ``physics_randomization.floor_friction_mu`` and
  ``floor_restitution`` for its ``<surface>`` block.
- Walls and obstacles emit per-object ``<surface>`` blocks driven by
  ``obstacle_friction_mu`` / ``obstacle_restitution``.
- Per-material RGBA is sourced from the per-scene ``visual_randomization``
  overrides; landmark colors stay at the LANDMARK_PALETTE values so the task
  identity signal is preserved.
- The directional sun light writes its direction, diffuse, specular, and
  ambient values from ``visual_randomization.lighting``.

Camera extrinsic jitter is recorded in the manifest but not written into the
SDF, because the camera lives on the robot URDF (sourced from the platform
manifest) rather than the world.
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from lewm_worlds.manifest import (
    BoxObject,
    PhysicsRandomization,
    SceneManifest,
    manifest_sha256,
)
from lewm_worlds.randomization import BASE_PALETTE, LANDMARK_PALETTE


# Fallback defaults if a manifest is built without randomization (legacy paths).
_DEFAULT_FLOOR_RGBA = BASE_PALETTE["floor"]
_DEFAULT_LIGHT_DIRECTION = (-0.5, 0.1, -0.9)
_DEFAULT_LIGHT_DIFFUSE = (0.8, 0.8, 0.8)
_DEFAULT_LIGHT_SPECULAR = (0.2, 0.2, 0.2)
_DEFAULT_LIGHT_AMBIENT = (0.25, 0.25, 0.25)
_DEFAULT_PHYSICS_RAND = PhysicsRandomization(
    floor_friction_mu=1.0,
    floor_restitution=0.0,
    obstacle_friction_mu=0.85,
    obstacle_restitution=0.0,
)


def export_gazebo_sdf(manifest: SceneManifest, out_dir: Path) -> Path:
    """Write ``world.sdf`` and ``manifest.json`` for a canonical scene."""

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
    material_lookup = _build_material_lookup(manifest)
    physics_random = manifest.physics_randomization or _DEFAULT_PHYSICS_RAND

    light_block = _render_light(manifest)
    ground_block = _render_ground(material_lookup, physics_random)
    models = "\n".join(
        _render_box_model(obj, material_lookup, physics_random)
        for obj in manifest.static_objects
    )
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
{light_block}
{ground_block}
{models}
  </world>
</sdf>
"""


def _build_material_lookup(manifest: SceneManifest) -> dict[str, tuple[float, float, float, float]]:
    """Resolve material_id -> RGBA for the active scene.

    Resolution order:
      1. per-scene visual_randomization.material_overrides
      2. fixed LANDMARK_PALETTE
      3. BASE_PALETTE fallback
      4. neutral grey
    """

    lookup: dict[str, tuple[float, float, float, float]] = {}
    if manifest.visual_randomization is not None:
        for override in manifest.visual_randomization.material_overrides:
            lookup[override.material_id] = override.rgba
    for key, rgba in LANDMARK_PALETTE.items():
        lookup.setdefault(key, rgba)
    for key, rgba in BASE_PALETTE.items():
        lookup.setdefault(key, rgba)
    return lookup


def _rgba_str(rgba: tuple[float, float, float, float]) -> str:
    r, g, b, a = rgba
    return f"{r} {g} {b} {a}"


def _surface_block(friction_mu: float, restitution: float) -> str:
    """Render a ``<surface>`` element pinning friction + restitution.

    SDF's ODE friction is named ``mu`` (primary friction coefficient). Bullet
    Featherstone (used by this project's Gazebo audit oracle) reads the same
    ``ode.mu`` value when no Bullet-specific block is present.
    """

    return (
        "        <surface>"
        f"<friction><ode><mu>{friction_mu}</mu><mu2>{friction_mu}</mu2></ode></friction>"
        f"<bounce><restitution_coefficient>{restitution}</restitution_coefficient></bounce>"
        "</surface>"
    )


def _render_light(manifest: SceneManifest) -> str:
    if manifest.visual_randomization is not None:
        lighting = manifest.visual_randomization.lighting
        direction = lighting.direction
        diffuse = lighting.diffuse_rgb
        specular = lighting.specular_rgb
        ambient = lighting.ambient_rgb
    else:
        direction = _DEFAULT_LIGHT_DIRECTION
        diffuse = _DEFAULT_LIGHT_DIFFUSE
        specular = _DEFAULT_LIGHT_SPECULAR
        ambient = _DEFAULT_LIGHT_AMBIENT

    diffuse_str = f"{diffuse[0]} {diffuse[1]} {diffuse[2]} 1"
    specular_str = f"{specular[0]} {specular[1]} {specular[2]} 1"
    ambient_str = f"{ambient[0]} {ambient[1]} {ambient[2]} 1"
    direction_str = f"{direction[0]} {direction[1]} {direction[2]}"

    return f"""    <scene>
      <ambient>{ambient_str}</ambient>
      <background>0.7 0.7 0.7 1</background>
    </scene>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>{diffuse_str}</diffuse>
      <specular>{specular_str}</specular>
      <direction>{direction_str}</direction>
    </light>"""


def _render_ground(
    material_lookup: dict[str, tuple[float, float, float, float]],
    physics_random: PhysicsRandomization,
) -> str:
    floor_rgba = material_lookup.get("floor", _DEFAULT_FLOOR_RGBA)
    color = _rgba_str(floor_rgba)
    surface = _surface_block(
        friction_mu=physics_random.floor_friction_mu,
        restitution=physics_random.floor_restitution,
    )
    return f"""    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry>
{surface}
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry>
          <material><ambient>{color}</ambient><diffuse>{color}</diffuse></material>
        </visual>
      </link>
    </model>"""


def _render_box_model(
    obj: BoxObject,
    material_lookup: dict[str, tuple[float, float, float, float]],
    physics_random: PhysicsRandomization,
) -> str:
    x, y, z = obj.center_xyz_m
    sx, sy, sz = obj.size_xyz_m
    color = _rgba_str(material_lookup.get(obj.material_id, (0.5, 0.5, 0.5, 1.0)))
    surface = _surface_block(
        friction_mu=physics_random.obstacle_friction_mu,
        restitution=physics_random.obstacle_restitution,
    )
    return f"""    <model name="{escape(obj.object_id)}">
      <static>true</static>
      <pose>{x} {y} {z} {obj.roll_rad} {obj.pitch_rad} {obj.yaw_rad}</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
{surface}
        </collision>
        <visual name="visual">
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
          <material><ambient>{color}</ambient><diffuse>{color}</diffuse></material>
        </visual>
      </link>
    </model>"""
