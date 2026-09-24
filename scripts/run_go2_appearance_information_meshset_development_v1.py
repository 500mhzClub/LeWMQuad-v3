"""Matched-pose actual renders, not robot execution or a replayed B outcome."""
from copy import deepcopy
import json
import math
import time

import cv2
import numpy as np
from PIL import Image

from lewm.causal_depth_observation_development import FOCAL,INTRINSICS,from_native_depth
from lewm.physical_execution_development import rotation_xyzw
from lewm.rgbd_correspondence_motion_development import RGBDCorrespondenceMotion
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.visual_surface_depth_evaluation_development import expected_optical_depth
from lewm_genesis.appearance_surface_development import ARMS,APPEARANCE_SEED,CELL_M,surfaces,triangle_identity
from lewm_genesis.appearance_meshset_development import visual_morph
from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import B,verify as verify_previous
from scripts.run_go2_appearance_information_development_v1 import OUTPUT as PREVIOUS
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.single_sample_rgbd_session_development import sampling_readback
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT=ROOT/'.generated/go2_appearance_information_meshset_development_v1_attempt_001'
PROTOCOL='docs/go2_appearance_information_meshset_development_v1_2026-09-06.md'
SEEDS=('scripts/run_go2_appearance_information_meshset_development_v1.py',
       'lewm/tests/test_appearance_meshset_development.py',PROTOCOL)
IDENTITIES={'launch.json':'bfb933ccaf1eeece784e6d35c0a0e48519662c28f53ebcade0b1d0474b790362',
    'failure.json':'bd7ee928e917cf98270b388fae5f3afff2b9215c496725a645ece8840667d53e',
    'neutral/ground_visual.ply':'24093742eea19845a24eb2175b6b1c66c443012ae355ed659ce3a6c6674804be'}
FRAME_INDICES=tuple(range(25,39))  # B 4.0s anchor plus all 13 depth-weak intervals.


def preflight():
    bindings={str((PREVIOUS/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()};verify_bindings(bindings)
    old=read_json(PREVIOUS,'launch.json');verify_previous(old)
    inputs=old['input_sha256']|bindings
    sources=discover_sources(SEEDS,old['source_sha256'])
    launch=old|dict(source_sha256=sources,input_sha256=inputs,arms=list(ARMS),appearance_seed=APPEARANCE_SEED,
        cell_m=CELL_M,source_camera_indices=list(FRAME_INDICES),
        scope='new appearance-only actual rendered sensor counterfactual; no physics, robot, original controller resume or navigation')
    verify_previous(launch);return launch


def boxes_from_recording():
    rows=read_json(B,'static_objects.json')
    return [dict(wall_id=r['native_name'],centre_xyz=r['pack_object']['center_xyz_m'],
        size_xyz=r['pack_object']['size_xyz_m'],yaw_rad=r['pack_object']['yaw_rad']) for r in rows]


def build(gs,arm,directory,boxes):
    scene=gs.Scene(sim_options=gs.options.SimOptions(dt=.002,gravity=(0.,0.,-9.81)),
                   show_viewer=False,renderer=gs.renderers.Rasterizer())
    physical=[scene.add_entity(gs.morphs.Plane(visualization=False),material=gs.materials.Rigid(friction=1.),name='ground_plane')]
    for box in boxes:
        physical.append(scene.add_entity(gs.morphs.Box(pos=box['centre_xyz'],size=box['size_xyz'],
            euler=(0.,0.,math.degrees(box['yaw_rad'])),fixed=True,visualization=False),
            material=gs.materials.Rigid(friction=1.),name=box['wall_id']))
    visual=[];expected={}
    for name,mesh in surfaces(boxes,arm):
        path=directory/(name+'.ply')
        morph=visual_morph(gs,path,mesh)
        expected[name]=triangle_identity(mesh.vertices.astype(np.float32),mesh.faces)
        visual.append(scene.add_entity(morph,name=name))
    camera=scene.add_camera(res=(640,480),pos=(0.,0.,.3),lookat=(1.,0.,.3),up=(0.,0.,1.),
        fov=math.degrees(2*math.atan(240/FOCAL)),near=.05,far=200.,GUI=False)
    scene.build(n_envs=1)
    identity=native_identity(physical,visual,expected,boxes)
    return scene,camera,physical,visual,expected,identity


def native_identity(physical,visual,expected,boxes):
    actual=[]
    for i,entity in enumerate(physical):
        assert len(entity.geoms)==1 and not entity.vgeoms and entity.morph.collision
        row=capture_native_robot_geometry(entity)[0]
        if i==0:
            assert row['geom_type']=='PLANE'
            np.testing.assert_allclose(row['position_world_m'],[0,0,0],atol=1e-9,rtol=0)
        else:
            box=boxes[i-1];assert row['geom_type']=='BOX' and entity.morph.fixed
            np.testing.assert_allclose(row['data'][:3],box['size_xyz'],atol=1e-8,rtol=0)
            np.testing.assert_allclose(row['position_world_m'],box['centre_xyz'],atol=1e-6,rtol=0)
        actual.append(row)
    witnesses=[]
    for entity in visual:
        assert not entity.geoms and not entity.morph.collision and entity.morph.fixed and len(entity.vgeoms)==1
        vg=entity.vgeoms[0];mesh=vg.get_trimesh()
        p=vg.get_pos().detach().cpu().numpy().reshape(3);q=vg.get_quat().detach().cpu().numpy().reshape(4)
        np.testing.assert_allclose(p,[0,0,0],atol=1e-9,rtol=0);np.testing.assert_allclose(q,[1,0,0,0],atol=1e-9,rtol=0)
        got=triangle_identity(mesh.vertices,mesh.faces);wanted=expected[entity.name]
        assert got['triangle_multiset_sha256']==wanted['triangle_multiset_sha256']
        np.testing.assert_allclose(got['area_m2'],wanted['area_m2'],atol=1e-6,rtol=0)
        witnesses.append(dict(name=entity.name,collision_geometries=0,geometry=got))
    return dict(physical_geometries=actual,visual_surfaces=witnesses,robot_present=False,physics_stepped=False)


def collect(arm,arm_index,boxes):
    import genesis as gs
    directory=OUTPUT/arm;directory.mkdir();scene=None;predictions=[];captures=[]
    try:
        scene,camera,physical,visual,expected,identity=build(gs,arm,directory,boxes)
        write_json(directory/'native_identity.json',identity)
        source_cameras=read_json(B,'camera_audit.json');observer=RGBDCorrespondenceMotion()
        for index,frame in enumerate(FRAME_INDICES):
            transform=np.asarray(source_cameras[frame]['world_from_optical'])
            pos=transform[:3,3];forward=transform[:3,2];up=-transform[:3,1]
            camera.set_pose(pos=pos,lookat=pos+forward,up=up);before=int(scene.t)
            start=time.perf_counter()
            rgb_out=camera.render(rgb=True,depth=False,segmentation=False,normal=False)
            depth_out=camera.render(rgb=False,depth=True,segmentation=False,normal=False)
            rgb=np.asarray(rgb_out[0]);native=np.asarray(depth_out[1])
            if rgb.ndim==4:rgb=rgb[0]
            if native.ndim==3:native=native[0]
            rgb=rgb[:,:,:3]
            assert rgb.shape==(480,640,3) and rgb.dtype==np.uint8 and native.shape==(480,640) and native.dtype==np.float32
            assert before==int(scene.t)==0
            np.testing.assert_allclose(camera.transform,transform,atol=1e-6,rtol=0)
            np.testing.assert_allclose(camera.intrinsics,INTRINSICS,atol=1e-7,rtol=0)
            sample=sampling_readback(camera)
            assert sample==dict(draw_framebuffer_is_single_sample_target=True,draw_framebuffer_is_multisample_target=False,
                samples=0,sample_buffers=0,multisample_enabled=False,pixel_scale=1)
            render_ms=1000*(time.perf_counter()-start)
            Image.fromarray(rgb).save(directory/f'rgb_{index:04d}.png')
            np.savez_compressed(directory/f'depth_{index:04d}.npz',optical_depth_m=native)
            reference=expected_optical_depth(boxes,transform,stride=8,floor_z_m=0.)
            true=reference['expected_depth_m'];good=reference['surface_interior']&(true>.22)&(true<4.98)
            error=np.abs(native[np.ix_(reference['rows'],reference['columns'])][good]-true[good])
            assert len(error)>1000 and np.isfinite(error).all() and error.max()<=.001
            captures.append(dict(source_B_frame=frame,world_from_optical=transform.tolist(),scene_steps=0,
                render_wall_ms=render_ms,sampling=sample,interior_depth_rays=len(error),maximum_depth_error_m=float(error.max())))
            policy,_=load_rgbd_observation(B,frame);fast=load_fast_packet(B,frame)
            # New rendered counterfactual identity; original files/episodes unchanged.
            policy=deepcopy(policy);fast=deepcopy(fast);policy['image']['rgb']=rgb.copy()
            policy['sensor_state']['identity']=(2,arm_index,0);fast['identity']=(2,arm_index,0)
            now=policy['sensor_state']['decision_ns']
            depth=from_native_depth(native,policy,measured_ns=now,available_ns=now,now_ns=now)
            start=time.perf_counter();row=observer.observe(policy,depth,fast,now_ns=now)
            predictions.append(row|dict(observer_wall_ms=1000*(time.perf_counter()-start),source_B_frame=frame))
        write_json(directory/'sensor_predictions.json',predictions);write_json(directory/'camera_evaluation.json',captures)
        terminal=native_identity(physical,visual,expected,boxes);assert terminal==identity
        write_json(directory/'terminal_native_identity.json',terminal)
        return identity
    finally:
        if scene is not None:scene.destroy()


def score(arm):
    predictions=read_json(OUTPUT/arm,'sensor_predictions.json');cameras=read_json(B,'camera_audit.json');rows=[]
    with np.load(B/'physics_trace.npz',allow_pickle=False) as raw:
        for previous,current in zip(predictions[:-1],predictions[1:],strict=True):
            i,j=[cameras[r['source_B_frame']]['physical_sample_index'] for r in (previous,current)]
            a,b=raw['base_pose_world'][i],raw['base_pose_world'][j]
            actual=rotation_xyzw(a[3:]).T@(b[:3]-a[:3]);pred=current['motion']['translation_previous_body_m']
            rows.append(dict(source_B_frame=current['source_B_frame'],status=current['motion']['status'],
                translation_error_m=float(np.linalg.norm(np.asarray(pred)-actual)) if pred is not None else None,
                keypoints=current['motion']['current_keypoints'],matches=current['motion']['mutual_ratio_matches'],
                lifted=current['motion']['lifted_matches'],inliers=current['motion']['inliers'],
                observer_wall_ms=current['observer_wall_ms']))
    return rows


def main():
    if OUTPUT.exists():raise ValueError('fresh fixed appearance assay output only; no retry')
    cv2.setNumThreads(1);launch=preflight();OUTPUT.mkdir();write_json(OUTPUT/'launch.json',launch)
    try:
        initialize_genesis(backend='cpu',seed=2026090605,logging_level='warning');boxes=boxes_from_recording();identities={}
        for i,arm in enumerate(ARMS):
            identities[arm]=collect(arm,i,boxes)
            assert identities[arm]==identities[ARMS[0]]
            print('APPEARANCE_RENDER_COMPLETE '+arm,flush=True)
        scores={arm:score(arm) for arm in ARMS}
        summaries={arm:dict(pairs=len(rows),accepted=sum(r['translation_error_m'] is not None for r in rows),
            maximum_error_m=max((r['translation_error_m'] for r in rows if r['translation_error_m'] is not None),default=None),
            mean_observer_wall_ms=float(np.mean([r['observer_wall_ms'] for r in rows]))) for arm,rows in scores.items()}
        verify_previous(launch);write_json(OUTPUT/'scored_pairs.json',scores)
        names=['scored_pairs.json']
        for arm in ARMS:
            names.extend(arm+'/'+n for n in ('native_identity.json','terminal_native_identity.json','sensor_predictions.json','camera_evaluation.json'))
            names.extend(arm+'/'+name+'.ply' for name in ['ground_visual',*[b['wall_id']+'_visual' for b in boxes]])
            names.extend(f'{arm}/{kind}_{i:04d}.{suffix}' for i in range(len(FRAME_INDICES)) for kind,suffix in (('rgb','png'),('depth','npz')))
        result=dict(status='MATCHED_APPEARANCE_RENDER_ASSAY_COMPLETE',summaries=summaries,
            environment_collision_and_visual_geometry_identical_across_arms=True,physics_steps=0,
            independent_motion_validation=False,original_B_result_changed=False,navigation_qualified=False,
            artifact_sha256={name:digest(OUTPUT/name) for name in names})
        write_json(OUTPUT/'result.json',result);print(json.dumps(summaries),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_APPEARANCE_ASSAY_FAILURE',error=repr(error)));raise
    finally:shutdown_genesis()


if __name__=='__main__':main()
