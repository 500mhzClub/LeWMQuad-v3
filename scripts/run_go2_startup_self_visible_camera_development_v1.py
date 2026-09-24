"""Fixed fitting-posture camera design; actual visual meshes, zero actuation."""
import json
import math
import importlib.metadata

import numpy as np
from PIL import Image

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_depth_observation_development import FOCAL,INTRINSICS
from lewm.longer_motion_collection_development import specification
from lewm.physical_execution_development import rotation_xyzw
from lewm.startup_camera_design_development import CANDIDATES,body_from_optical,classify_depth,footprint_visibility
from lewm_genesis.appearance_meshset_development import visual_morph
from lewm_genesis.optical_camera_readback_development import check_optical_pose
from lewm_genesis.rgbd_motion_scene_development import independently_seeded_surfaces
from lewm_genesis.scene_builder import _import_genesis,initialize_genesis,shutdown_genesis
from scripts.analyze_go2_ground_plane_development_v1 import URDF,verify_bindings
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_robot_visual_assets_development import robot_mesh,verify_assets
from scripts.startup_source_inventory_development import discover_sources

OUTPUT=ROOT/'.generated/go2_startup_self_visible_camera_development_v1_attempt_001'
PREVIOUS=ROOT/'.generated/go2_bounded_rotation_geometry_adapter_v1_attempt_001'
INPUT=ROOT/'.generated/go2_longer_observed_floor_motion_development_v1_attempt_001/fit/physics_trace.npz'
PROTOCOL='docs/go2_startup_self_visible_camera_development_v1_2026-09-06.md'
IDENTITIES={'launch.json':'b907e7075397a1ca1671ff3866028f265a633f4d775f51592c53452f80673978',
 'result.json':'1984b2dcd11965e12012418a85cd5b9a1e41ac0acdba9d982e476fb236eae202',
 'corner_and_cell_audit_launch.json':'e2f69d0231c83cfeae544059d5e053b792a1bbe5c1fe09486f94032920082f79',
 'corner_and_cell_audit.json':'3bf1c86951c340bf7c2df1625507903a45fa8b5a557a543d5e5c2d74591c845f'}


def preflight():
    bindings={str((PREVIOUS/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()}; verify_bindings(bindings)
    old=read_json(PREVIOUS,'launch.json'); verify(old)
    audit=read_json(PREVIOUS,'corner_and_cell_audit_launch.json')
    verify_bindings(audit['source_sha256']|audit['input_sha256'])
    sources=discover_sources((PROTOCOL,'scripts/run_go2_startup_self_visible_camera_development_v1.py',
        'lewm/tests/test_startup_camera_design_development.py'),old['source_sha256']|audit['source_sha256'])
    launch=old|dict(source_sha256=sources,input_sha256=old['input_sha256']|audit['input_sha256']|bindings,
        visual_asset_sha256=verify_assets(),candidates=CANDIDATES,source_sample=749,
        package_versions={n:importlib.metadata.version(n) for n in ('trimesh','pycollada','genesis-world')},
        scope='new hypothetical startup camera render-only fitting design; no actuation or navigation')
    if str(INPUT.relative_to(ROOT)) not in launch['input_sha256']: raise ValueError('recorded configuration must be bound')
    verify(launch); return launch


def configuration():
    with np.load(INPUT,allow_pickle=False) as raw:
        q=raw['joint_position'][749].copy(); pose=raw['base_pose_world'][749].copy()
    T=np.eye(4); T[:3,:3]=rotation_xyzw(pose[3:]); T[:3,3]=pose[:3]
    return q,T


def render_scene(gs,directory,surfaces,robot,Tworld):
    directory.mkdir(); scene=gs.Scene(sim_options=gs.options.SimOptions(dt=.002,gravity=(0,0,-9.81)),
        show_viewer=False,renderer=gs.renderers.Rasterizer()); names=[]; rasters={}; camera_rows={}
    try:
        meshes=surfaces+([('robot_visual',robot)] if robot is not None else [])
        for name,mesh in meshes:
            scene.add_entity(visual_morph(gs,directory/(name+'.ply'),mesh),name=name); names.append(name+'.ply')
        camera=scene.add_camera(res=(640,480),pos=(0,0,1),lookat=(1,0,1),up=(0,0,1),
            fov=math.degrees(2*math.atan(240/FOCAL)),near=.05,far=200.,GUI=False)
        scene.build(n_envs=1)
        for name,_,_ in CANDIDATES:
            T=Tworld@body_from_optical(name); p=T[:3,3]
            camera.set_pose(pos=p,lookat=p+T[:3,2],up=-T[:3,1])
            actual=check_optical_pose(camera.transform,T)
            np.testing.assert_allclose(camera.intrinsics,INTRINSICS,atol=1e-7,rtol=0)
            if int(scene.t)!=0: raise ValueError('render-only zero-step required')
            rendered=camera.render(rgb=True,depth=True,segmentation=False,normal=False)
            rgb,depth=np.asarray(rendered[0]),np.asarray(rendered[1])
            if rgb.shape==(1,480,640,3): rgb=rgb[0]
            if depth.shape==(1,480,640): depth=depth[0]
            if rgb.shape!=(480,640,3) or rgb.dtype!=np.uint8 or depth.shape!=(480,640) or depth.dtype!=np.float32:
                raise ValueError('native RGB/depth encoding differs')
            if int(scene.t)!=0 or rendered[2] is not None or rendered[3] is not None: raise ValueError('zero-step RGB/depth only')
            with (directory/(name+'.png')).open('xb') as f: Image.fromarray(rgb).save(f,format='PNG')
            with (directory/(name+'.npz')).open('xb') as f: np.savez_compressed(f,optical_depth_m=depth)
            names.extend((name+'.png',name+'.npz')); rasters[name]=depth.copy()
            camera_rows[name]=dict(world_from_optical=T.tolist(),native_converted_pose=actual.tolist(),
                intrinsics=np.asarray(camera.intrinsics).tolist(),near=camera.near,far=camera.far,physics_steps=int(scene.t))
            print('STARTUP_RENDER_COMPLETE',directory.name,name,flush=True)
        write_json(directory/'cameras.json',camera_rows); names.append('cameras.json')
        return rasters,names
    finally: scene.destroy()


def score(visible,background,q,T):
    shapes=ArticulatedCollisionGeometry(URDF).supports(q,T[:3,:3])['shapes']; rows={}
    for name,_,_ in CANDIDATES:
        camera=T@body_from_optical(name); masks=classify_depth(visible[name],background[name],camera)
        footprints={s['shape_id']:footprint_visibility(np.asarray(s['lower'])+T[:3,3],
            np.asarray(s['upper'])+T[:3,3],camera,masks) for s in shapes}
        rows[name]=dict(pixel_counts={k:int(v.sum()) for k,v in masks.items()},footprints=footprints,
            complete_frustum_shapes=sum(r['complete_frustum'] for r in footprints.values()),
            complete_floor_rectangle_shapes=sum(r['complete_rectangle_floor'] for r in footprints.values()),
            observed_samples=sum(sum(r['sampled_floor_visibility']) for r in footprints.values()),total_samples=675)
    pair={k:(np.asarray(rows['left_outboard_down90']['footprints'][k]['sampled_floor_visibility'])|
        np.asarray(rows['right_outboard_down90']['footprints'][k]['sampled_floor_visibility'])).tolist() for k in rows['forward']['footprints']}
    return dict(cameras=rows,outboard_pair_sampled_union=pair,outboard_pair_observed_samples=sum(sum(v) for v in pair.values()),
        total_samples=675,sampled_union_is_not_continuous_coverage=True,world_axis_collision_support=shapes,
        world_from_body=T.tolist(),joints=q.tolist(),physics_steps=0,navigation_qualified=False,goal_achieved=False)


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive startup design output required')
    launch=preflight(); OUTPUT.mkdir(); write_json(OUTPUT/'launch.json',launch)
    try:
        q,T=configuration(); robot,witnesses=robot_mesh(q,T[:3,:3],T[:3,3])
        write_json(OUTPUT/'visual_instances.json',witnesses); names=['visual_instances.json']
        spec=specification('fit'); surfaces=independently_seeded_surfaces(spec['geometry']['wall_boxes'],spec['appearance_arm'],spec['appearance_seed'])
        initialize_genesis(backend='cpu',seed=2026090633,logging_level='warning'); gs=_import_genesis()
        visible,files=render_scene(gs,OUTPUT/'visible',surfaces,robot,T); names.extend('visible/'+n for n in files)
        background,files=render_scene(gs,OUTPUT/'background',surfaces,None,T); names.extend('background/'+n for n in files)
        for name,_ in surfaces:
            if digest(OUTPUT/'visible'/(name+'.ply'))!=digest(OUTPUT/'background'/(name+'.ply')):
                raise ValueError('paired background mesh identity mismatch')
        result=score(visible,background,q,T); verify(launch)
        if verify_assets()!=launch['visual_asset_sha256']: raise ValueError('visual asset change')
        result|=dict(status='STARTUP_CAMERA_DESIGN_COMPLETE',artifact_sha256={n:digest(OUTPUT/n) for n in names})
        write_json(OUTPUT/'result.json',result)
        print(json.dumps({k:{n:v for n,v in row.items() if n!='footprints'} for k,row in result['cameras'].items()}),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_STARTUP_DESIGN_FAILURE',reason=repr(error))); raise
    finally: shutdown_genesis()


if __name__=='__main__': main()
