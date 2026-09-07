"""New bounded/aligned Go2 scene builder; frozen original remains unchanged."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from lewm.causal_depth_observation_development import FOCAL
from lewm_genesis.scene_builder import (
    SceneBuild, ScenePack, _import_genesis, initialize_genesis, _floor_material,
    _obstacle_material, _material_lookup, select_scene_textures,
    _diffuse_texture_surface, _surface_for, category_for_kind, cached_box_obj,
    _initial_camera_pose_world, genesis_vertical_fov_deg,
)
from lewm_genesis.floor_extent_precision_development import read_extent_identity


@dataclass
class BoundedSceneBuild(SceneBuild):
    collision_floor: Any
    visual_floor: Any
    floor_domain: dict


def floor_domain(pack):
    """Scene-construction/evaluation domain, never a policy observation."""
    bounds = np.asarray(pack.world_bounds_xy_m, float)
    mount = np.asarray(pack.camera.xyz_body_m, float)
    spawn = np.asarray(pack.robot.spawn_xyz_m, float)
    if (bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0])
            or mount.shape != (3,) or not np.isfinite(mount).all()
            or spawn.shape != (3,) or not np.isfinite(spawn).all()
            or np.any(spawn[:2] < bounds[0]) or np.any(spawn[:2] > bounds[1])):
        raise ValueError('finite declared body workspace, mount and in-workspace spawn required')
    camera = pack.camera
    if (tuple(camera.native_resolution) != (640, 480) or camera.fov_axis != 'horizontal'
            or camera.fov_deg != 78.323 or camera.near_m != .05 or camera.far_m != 200.
            or not np.array_equal(camera.rpy_body_rad, [0., 0., 0.])
            or getattr(pack, 'camera_extrinsic_jitter', None) is not None):
        raise ValueError('fixed reviewed RGBD calibration without jitter required')
    radius = float(np.linalg.norm(mount) + 5. * np.sqrt(1. + (320. / FOCAL)**2 + (240. / FOCAL)**2))
    margin = 16. - float(np.abs(bounds).max()) - radius
    if margin <= .01:
        raise ValueError('declared workspace plus all5m optical rays exceeds finite floor domain')
    return {'body_workspace_xy_m': bounds.tolist(), 'visual_half_extent_m': 16.,
            'camera_mount_plus_ray_radius_m': radius, 'minimum_declared_support_margin_m': margin,
            'scope': 'scene/evaluation-only finite support; not policy input or navigation qualification'}


def check_capture_domain(build, world_from_optical):
    """Fail acquisition outside actual finite visual support, never fill depth."""
    T = np.asarray(world_from_optical, float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError('finite actual camera transform required')
    R = T[:3, :3]
    if (not np.array_equal(T[3], [0., 0., 0., 1.]) or not np.allclose(R.T @ R, np.eye(3), atol=1e-7, rtol=0)
            or abs(np.linalg.det(R) - 1) > 1e-7):
        raise ValueError('proper actual optical pose required')
    rays = np.array([[x/FOCAL, y/FOCAL, 1.] for x in (-320., 320.) for y in (-240., 240.)])
    vertices = np.concatenate([T[:3, 3] + d * (rays @ R.T) for d in (.2, 5.)])
    if np.any(np.abs(vertices[:, :2]) >= 15.99):
        raise ValueError('actual capture frustum exceeds reviewed finite visual support')
    return {'maximum_frustum_abs_xy_m': np.abs(vertices[:, :2]).max(axis=0).tolist(),
            'finite_visual_support_verified': True, 'navigation_qualified': False}


def build_scene_from_pack(
    pack: ScenePack,
    *,
    n_envs: int,
    backend: str = "auto",
    show_viewer: bool = False,
    render_robot: bool = True,
    apply_textures: bool = False,
    batched_camera: bool = False,
) -> SceneBuild:
    """Build a Genesis scene with ``n_envs`` parallel envs from a ``ScenePack``.

    ``render_robot=False`` keeps the robot fully present in physics but skips its
    visual meshes from camera renders, so egocentric captures do not contain the
    robot body when the safety retraction pulls the camera back inside the body
    envelope.
    """

    if n_envs != 1 or backend != 'cpu' or batched_camera or show_viewer:
        raise ValueError('reviewed one-environment offscreen CPU scene required')
    domain = floor_domain(pack)
    gs = _import_genesis()
    initialize_genesis(backend=backend, seed=pack.physics_seed)

    scene_kwargs: dict[str, Any] = {
        "sim_options": gs.options.SimOptions(
            dt=float(pack.timing.physics_dt_s),
            gravity=(0.0, 0.0, -9.81),
        ),
        "show_viewer": bool(show_viewer),
        "renderer": gs.renderers.Rasterizer(),
    }
    # Batched rendering: env_separate_rigid makes a single (env_idx=None) camera
    # render ALL envs in one call (rgb shape (n_envs, H, W, 3)), so a scene's
    # parallel rollout streams render together instead of one-per-call.
    if batched_camera and int(n_envs) > 1:
        scene_kwargs["vis_options"] = gs.options.VisOptions(
            env_separate_rigid=True,
            rendered_envs_idx=list(range(int(n_envs))),
        )
    scene = gs.Scene(**scene_kwargs)

    floor_material = _floor_material(gs, pack)
    obstacle_material = _obstacle_material(gs, pack)
    material_lookup = _material_lookup(pack)

    # Per-scene texture theme (data-spec §14 visuals). Render-only: rollouts
    # pass apply_textures=False so the physics path keeps fast box primitives.
    scene_textures: dict[str, str | None] = (
        select_scene_textures(visual_seed=pack.visual_seed, scene_id=pack.scene_id)
        if apply_textures
        else {}
    )

    floor_tex = scene_textures.get("floor")
    floor_surface = (_diffuse_texture_surface(gs, floor_tex) if floor_tex else None) or (
        _surface_for(gs, "floor", material_lookup)
    )

    plane_kwargs: dict[str, Any] = {}
    if floor_material is not None:
        plane_kwargs["material"] = floor_material
    if floor_surface is not None:
        plane_kwargs["surface"] = floor_surface
    collision_floor = scene.add_entity(gs.morphs.Plane(visualization=False), **plane_kwargs)
    visual_floor = scene.add_entity(gs.morphs.Plane(pos=(0., 0., .005), collision=False,
                                                   plane_size=(32., 32.)), **plane_kwargs)

    for obj in pack.static_objects:
        # Box euler is in degrees per Genesis morph contract; convert from
        # radians stored on the manifest.
        euler_deg = (
            math.degrees(float(obj.roll_rad)),
            math.degrees(float(obj.pitch_rad)),
            math.degrees(float(obj.yaw_rad)),
        )
        category = category_for_kind(obj.kind) if apply_textures else None
        tex_path = scene_textures.get(category) if category else None
        tex_surface = _diffuse_texture_surface(gs, tex_path) if tex_path else None

        entity_kwargs: dict[str, Any] = {"name": obj.object_id}
        if obstacle_material is not None:
            entity_kwargs["material"] = obstacle_material

        if tex_surface is not None:
            # Box primitives can't carry a texture; use a UV-mapped cube mesh
            # (collision preserved). OBJ is authored Z-up in metres.
            morph = gs.morphs.Mesh(
                file=cached_box_obj(obj.size_xyz_m),
                pos=obj.center_xyz_m,
                euler=euler_deg,
                fixed=True,
                collision=True,
                convexify=True,
                file_meshes_are_zup=True,
            )
            entity_kwargs["surface"] = tex_surface
        else:
            morph = gs.morphs.Box(
                pos=obj.center_xyz_m,
                size=obj.size_xyz_m,
                euler=euler_deg,
                fixed=True,
            )
            surface = _surface_for(gs, obj.material_id, material_lookup)
            if surface is not None:
                entity_kwargs["surface"] = surface
        scene.add_entity(morph, **entity_kwargs)

    # Genesis quaternion convention: wxyz (matches the scene manifest).
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(pack.robot.urdf_path),
            pos=pack.robot.spawn_xyz_m,
            quat=pack.robot.spawn_quat_wxyz,
            fixed=False,
            visualization=bool(render_robot),
            collision=True,
        ),
        name="go2",
    )

    cam_pos_world, cam_lookat_world = _initial_camera_pose_world(pack)
    camera = scene.add_camera(
        res=pack.camera.native_resolution,
        pos=cam_pos_world,
        lookat=cam_lookat_world,
        fov=genesis_vertical_fov_deg(pack.camera),
        near=float(pack.camera.near_m),
        far=float(pack.camera.far_m),
        GUI=False,
    )

    scene.build(n_envs=int(n_envs))
    read_extent_identity(collision_floor, visual_floor, 32.)
    return BoundedSceneBuild(scene=scene, robot=robot, camera=camera, pack=pack, n_envs=int(n_envs),
                             collision_floor=collision_floor, visual_floor=visual_floor, floor_domain=domain)
