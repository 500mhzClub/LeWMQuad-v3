"""Distinct union-boundary visuals; native collision boxes and5mm camera unchanged."""
from dataclasses import replace
from lewm_genesis.union_wall_surface_development import independently_seeded_union_surfaces
import math
import numpy as np
from lewm_genesis.rgbd_motion_scene_development import (
    AppearanceRobotBuild, independently_seeded_surfaces)
from lewm_genesis.appearance_surface_development import triangle_identity
from lewm_genesis.appearance_meshset_development import visual_morph
from lewm_genesis.bounded_scene_builder_development import floor_domain as old_floor_domain
from lewm_genesis.scene_builder import (
    _import_genesis, initialize_genesis, _floor_material, _obstacle_material,
    _initial_camera_pose_world, genesis_vertical_fov_deg)
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.run_go2_appearance_information_development_v1 import native_identity

RENDER_NEAR_M = .005


def floor_domain(pack):
    if pack.camera.near_m != RENDER_NEAR_M:
        raise ValueError('distinct renderer requires exactly5mm near plane')
    # Only the legacy domain validator receives a temporary calibration copy.
    # Its radius uses public5m rays and the unchanged mount; not the near plane.
    # Neither native construction nor persisted pack receives the legacy value.
    return old_floor_domain(replace(pack, camera=replace(pack.camera, near_m=.05)))


def build_scene_from_pack(pack, *, output, appearance_arm, appearance_seed,
                          n_envs=1, backend='cpu', show_viewer=False, render_robot=False):
    if n_envs!=1 or backend!='cpu' or show_viewer or render_robot:
        raise ValueError('one CPU offscreen robot-hidden development scene required')
    if not output.is_dir():raise ValueError('fresh declared output directory required')
    if any(o.roll_rad!=0 or o.pitch_rad!=0 for o in pack.static_objects):
        raise ValueError('upright box environment required; do not ignore roll/pitch')
    domain=floor_domain(pack)
    boxes=[dict(wall_id=o.object_id,centre_xyz=o.center_xyz_m,size_xyz=o.size_xyz_m,yaw_rad=o.yaw_rad) for o in pack.static_objects]
    # Construct/validate procedural meshes before native allocation.
    meshes=independently_seeded_union_surfaces(boxes,appearance_arm,appearance_seed)
    gs=_import_genesis(); initialize_genesis(backend=backend,seed=pack.physics_seed)
    scene=gs.Scene(sim_options=gs.options.SimOptions(dt=float(pack.timing.physics_dt_s),gravity=(0.,0.,-9.81)),
        show_viewer=False,renderer=gs.renderers.Rasterizer())
    try:
        floor_material=_floor_material(gs,pack); obstacle_material=_obstacle_material(gs,pack)
        floor=scene.add_entity(gs.morphs.Plane(visualization=False),material=floor_material,name='ground_plane')
        physical=[floor]
        for box in boxes:
            physical.append(scene.add_entity(gs.morphs.Box(pos=box['centre_xyz'],size=box['size_xyz'],
                euler=(0.,0.,math.degrees(box['yaw_rad'])),fixed=True,visualization=False),
                material=obstacle_material,name=box['wall_id']))
        robot=scene.add_entity(gs.morphs.URDF(file=str(pack.robot.urdf_path),pos=pack.robot.spawn_xyz_m,
            quat=pack.robot.spawn_quat_wxyz,fixed=False,visualization=False,collision=True),name='go2')
        visual=[]; expected={}
        for name,mesh in meshes:
            morph=visual_morph(gs,output/(name+'.ply'),mesh)
            expected[name]=triangle_identity(mesh.vertices.astype(np.float32),mesh.faces)
            visual.append(scene.add_entity(morph,name=name))
        pos,lookat=_initial_camera_pose_world(pack)
        camera=scene.add_camera(res=pack.camera.native_resolution,pos=pos,lookat=lookat,
            fov=genesis_vertical_fov_deg(pack.camera),near=float(pack.camera.near_m),far=float(pack.camera.far_m),GUI=False)
        scene.build(n_envs=1)
        identity=native_identity(physical,visual,expected,boxes)
        # The static helper's statement applies only to its entity list. This
        # scene DOES contain the real robot; never inherit its no-robot claim.
        identity=identity|dict(robot_present=True)
        robot_geometry=capture_native_robot_geometry(robot)
        if not robot_geometry or int(scene.t)!=0 or not robot.morph.collision or robot.morph.visualization:
            raise ValueError('actual collision-only Go2 geometry and zero-step construction required')
        return AppearanceRobotBuild(scene=scene,robot=robot,camera=camera,pack=pack,n_envs=1,
            collision_floor=floor,visual_floor=visual[0],floor_domain=domain,physical_environment=physical,
            visual_surfaces=visual,expected_visual_geometry=expected,native_environment_identity=identity,
            native_robot_geometry=robot_geometry)
    except Exception:
        scene.destroy()
        raise


