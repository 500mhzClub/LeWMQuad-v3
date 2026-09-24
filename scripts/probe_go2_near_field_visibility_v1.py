"""Fixed two-arm native camera bench; zero physics, preserve all clipping failures."""
from dataclasses import replace
import json
import shutil
import cv2
import numpy as np
import torch
from PIL import Image
from lewm.causal_depth_observation_development import INTRINSICS
from lewm.physical_semantics import world_from_optical
from lewm.physical_first_surface_depth_development import evaluate_visibility
from lewm.longer_motion_collection_development import specification,pack
from lewm_genesis.scene_loader import StaticObject
from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
from lewm_genesis.rgbd_motion_scene_development import build_scene_from_pack as legacy_build
from lewm_genesis.near_field_rgbd_scene_development import build_scene_from_pack as corrected_build
from lewm_genesis.optical_camera_readback_development import check_optical_pose
from lewm_genesis.bounded_scene_builder_development import check_capture_domain
from scripts.single_sample_rgbd_session_development import sampling_readback
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json,read_npz
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.navigation_artifact_root_development import BASE,create_output,verify_artifacts,validate_root

OUTPUT=BASE/'go2_near_field_visibility_probe_v1_attempt_001'
PILOT=BASE/'go2_independent_pulse_context_pilot_v1_attempt_001'
PROTOCOL='docs/go2_near_field_visibility_probe_v1_2026-09-06.md'
DISTANCES=(.004,.006,.020,.049,.051,.199,.201,1.,4.9)
ARMS=(('legacy',.05),('corrected',.005))
RESERVE=40*1024**3
BUDGET=256*1024**2


def boxes():
    return [dict(wall_id=name,centre_xyz=[x,0.,6.],size_xyz=[.08,12.,12.],yaw_rad=0.)
        for name,x in (('front',.6),('rear',1.8))]


def scene_pack(near):
    if near not in (.05,.005):raise ValueError('fixed two-arm renderer comparison required')
    base=pack(specification('fit'))
    objects=tuple(StaticObject(object_id=b['wall_id'],kind='wall',center_xyz_m=tuple(b['centre_xyz']),
        size_xyz_m=tuple(b['size_xyz']),yaw_rad=0.,material_id='wall') for b in boxes())
    return replace(base,scene_id='near-field-visibility-bench-v1',family='NATIVE_CAMERA_BENCH_ONLY',
        static_objects=objects,physics_seed=2026090901,visual_seed=2026090902,
        camera=replace(base.camera,near_m=near))


def preflight():
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive fixed bench; no retry/resume')
    verify_artifacts(PILOT,{'launch.json':'bb06bb68d6cc8702c224d1a5b78d5b4285b2c3dcf6636e9c029f425d0519319d'})
    old=read_json(PILOT,'launch.json');verify(old)
    sources=discover_sources((PROTOCOL,'scripts/probe_go2_near_field_visibility_v1.py',
        'lewm/tests/test_physical_first_surface_depth_development.py'),old['source_sha256'])
    launch={k:old[k] for k in ('input_sha256','native_sha256','native_geometry_sha256','opencv_binary_sha256','opencv_version','rules')}
    launch.update(source_sha256=sources,distances_m=list(DISTANCES),arms=dict(ARMS),boxes=boxes(),
        output_root=str(OUTPUT),physics_seed=2026090901,appearance_seed=2026090902,
        minimum_free_bytes=RESERVE,maximum_artifact_bytes=BUDGET,physics_steps=0,
        expected_captures=18,model_training=False,navigation_qualified=False)
    verify(launch)
    if shutil.disk_usage(BASE.parent).free<RESERVE+BUDGET:raise ValueError('bench storage reserve required')
    return launch


def capture_arm(label,near):
    directory=OUTPUT/label;directory.mkdir();visual=directory/'visual_meshes';visual.mkdir()
    build=None;rows=[]
    initialize_genesis(backend='cpu',seed=2026090901,logging_level='warning')
    try:
        builder=legacy_build if label=='legacy' else corrected_build
        build=builder(scene_pack(near),output=visual,appearance_arm='distinctive',appearance_seed=2026090902)
        camera=build.camera
        write_json(directory/'native_identity.json',dict(environment=build.native_environment_identity,
            robot=build.native_robot_geometry,physics_steps=int(build.scene.t)))
        for i,distance in enumerate(DISTANCES):
            if shutil.disk_usage(BASE.parent).free<RESERVE:raise ValueError('bench storage reserve exhausted')
            position=np.array([.56-distance,0.,6.]);forward=np.array([1.,0.,0.]);up=np.array([0.,0.,1.])
            T=world_from_optical(position,forward,up);check_capture_domain(build,T)
            camera.set_pose(pos=position,lookat=position+forward,up=up);check_optical_pose(camera.transform,T)
            np.testing.assert_allclose(camera.intrinsics,INTRINSICS,atol=1e-7,rtol=0)
            if (camera.near!=near or camera.far!=200. or tuple(camera.res)!=(640,480)
                    or camera._raytracer is not None or camera._batch_renderer is not None):
                raise ValueError('exact native camera contract required')
            before=int(build.scene.t);pose_before=np.array(camera.transform,copy=True)
            rgb_output=camera.render(rgb=True,depth=False,segmentation=False,normal=False)
            depth_output=camera.render(rgb=False,depth=True,segmentation=False,normal=False)
            if before!=0 or int(build.scene.t)!=0 or not np.array_equal(pose_before,camera.transform):
                raise ValueError('no physics or pose change between RGB/depth allowed')
            sample=sampling_readback(camera)
            if sample!=dict(draw_framebuffer_is_single_sample_target=True,draw_framebuffer_is_multisample_target=False,
                    samples=0,sample_buffers=0,multisample_enabled=False,pixel_scale=1):
                raise ValueError('exact single-sample native framebuffer required')
            rgb=np.asarray(rgb_output[0]);native=np.asarray(depth_output[1])
            if rgb.ndim==4 and rgb.shape[0]==1:rgb=rgb[0]
            if native.ndim==3 and native.shape[0]==1:native=native[0]
            rgb=rgb[...,:3]
            if rgb.shape!=(480,640,3) or rgb.dtype!=np.uint8 or native.shape!=(480,640) or native.dtype!=np.float32:
                raise ValueError('exact native image formats required')
            Image.fromarray(rgb).save(directory/f'rgb_{i:04d}.png')
            np.savez_compressed(directory/f'native_depth_{i:04d}.npz',optical_depth_m=native)
            report=evaluate_visibility(native,boxes(),T,render_near_m=near)
            row=dict(index=i,distance_m=distance,world_from_optical=T.tolist(),near_m=near,far_m=camera.far,
                intrinsics=np.asarray(camera.intrinsics).tolist(),sampling_readback=sample,physics_steps_before_after=[before,int(build.scene.t)],
                centre_depth_m=float(native[240,320]),public_valid_pixels=int((np.isfinite(native)&(native>=.2)&(native<=5.)).sum()),
                evaluation=report,expected_pass=distance>near)
            rows.append(row);write_json(directory/f'frame_{i:04d}.json',row)
            print('NEAR_FIELD_CAPTURE',label,i,distance,row['centre_depth_m'],report,flush=True)
        return rows
    finally:
        if build is not None:build.scene.destroy()
        shutdown_genesis()


def artifact_names():
    return [label+'/'+name for label,_ in ARMS for name in
        (['native_identity.json','visual_meshes/ground_visual.ply','visual_meshes/front_visual.ply','visual_meshes/rear_visual.ply']+
        [f'{prefix}_{i:04d}.{suffix}' for i in range(len(DISTANCES))
            for prefix,suffix in (('rgb','png'),('native_depth','npz'),('frame','json'))])]


def verify_saved(bindings):
    verify_artifacts(OUTPUT,bindings);reports=[]
    for label,near in ARMS:
        for i,distance in enumerate(DISTANCES):
            row=read_json(OUTPUT/label,f'frame_{i:04d}.json')
            T=world_from_optical([.56-distance,0.,6.],[1.,0.,0.],[0.,0.,1.])
            np.testing.assert_array_equal(row['world_from_optical'],T)
            if (row['near_m']!=near or row['distance_m']!=distance or row['expected_pass']!=(distance>near)
                    or row['physics_steps_before_after']!=[0,0]):raise ValueError('fixed capture identity changed')
            native=read_npz(OUTPUT/label,f'native_depth_{i:04d}.npz')['optical_depth_m']
            report=evaluate_visibility(native,boxes(),T,render_near_m=near)
            if json.dumps(row['evaluation'],sort_keys=True)!=json.dumps(report,sort_keys=True):raise ValueError('saved native visibility score mismatch')
            # Independent planar bench oracle: both walls cover the whole frustum.
            correct=float(np.abs(native.astype(np.float64)-distance).max())<=.001
            if correct!=report['passes_sampled_physical_visibility'] and distance>near:
                raise ValueError('full-image planar oracle and sampled visibility disagree')
            reports.append(dict(arm=label,distance_m=distance,report=report,
                full_image_max_error_m=float(np.abs(native.astype(np.float64)-distance).max()),
                expected_outcome_observed=report['passes_sampled_physical_visibility']==(distance>near)))
    verify_artifacts(OUTPUT,bindings);return reports


def main():
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    launch=preflight();create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    try:
        for label,near in ARMS:verify(launch);capture_arm(label,near)
        names=artifact_names();bindings={n:digest(OUTPUT/n) for n in names}
        used=sum((OUTPUT/n).stat().st_size for n in names)+(OUTPUT/'launch.json').stat().st_size
        if used>BUDGET:raise ValueError('bench artifact budget exceeded')
        rows=verify_saved(bindings);verify(launch)
        result=dict(status='NATIVE_NEAR_FIELD_BENCH_COMPLETE',all_expected_outcomes_observed=all(r['expected_outcome_observed'] for r in rows),
            rows=rows,artifact_sha256=bindings,artifact_bytes=used,physics_steps=0,captures=18,
            navigation_qualified=False,hardware_calibrated=False,goal_achieved=False)
        write_json(OUTPUT/'result.json',result);print('NEAR_FIELD_BENCH_COMPLETE',result['all_expected_outcomes_observed'],flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_NATIVE_NEAR_FIELD_BENCH_FAILURE',reason=repr(error)))
        raise


if __name__=='__main__':main()
