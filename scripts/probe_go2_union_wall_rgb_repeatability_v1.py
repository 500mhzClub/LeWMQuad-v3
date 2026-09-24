"""Two independent static native union-wall builds at two fixed recorded poses."""
import json
import shutil
import cv2
import numpy as np
import torch
from PIL import Image
from lewm.causal_depth_observation_development import INTRINSICS
from lewm.physical_first_surface_depth_development import evaluate_visibility
from lewm_genesis.union_wall_rgbd_scene_development import build_scene_from_pack
from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
from lewm_genesis.bounded_scene_builder_development import check_capture_domain
from lewm_genesis.optical_camera_readback_development import check_optical_pose
from scripts.single_sample_rgbd_session_development import sampling_readback
from scripts.independent_layout_batch_development import load_inventory,output_root
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json,read_npz
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.navigation_artifact_root_development import BASE,validate_root,create_output,verify_artifacts

OUTPUT=BASE/'go2_union_wall_rgb_repeatability_probe_v1_attempt_001'
PROTOCOL='docs/go2_union_wall_rgb_repeatability_probe_v1_2026-09-06.md'
INPUT=output_root('l00')
IDS={'launch.json':'1d46515179b91d7134c83b6247f0cc22db1de786e3a99a1db9bde82ca85fb218',
    'episode_000_commit.json':'d2a0150ca7552f228f5c3a0d75f48eb43cdaa62e04d57350f9b40c6bf1eab25b'}
RESERVE=40*1024**3
BUDGET=256*1024**2


def preflight():
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive static bench, no retry/resume')
    verify_artifacts(INPUT,IDS);old=read_json(INPUT,'launch.json');verify(old)
    commit=read_json(INPUT,'episode_000_commit.json');trial=old['planned_trials'][0]
    assert commit['trial']==trial
    names=[trial+'/'+n for n in ('specification.json','camera_audit.json')]
    bound=IDS|{n:commit['artifact_sha256'][n] for n in names};verify_artifacts(INPUT,bound)
    inv=load_inventory();spec=inv.specification(trial)
    if read_json(INPUT/trial,'specification.json')!=spec:raise ValueError('exact recorded scene required')
    cameras=read_json(INPUT/trial,'camera_audit.json');poses=[cameras[i]['world_from_optical'] for i in (0,8)]
    sources=discover_sources((PROTOCOL,'scripts/probe_go2_union_wall_rgb_repeatability_v1.py',
        'lewm/tests/test_union_wall_surface_development.py'),old['source_sha256'])
    launch={k:old[k] for k in ('input_sha256','native_sha256','native_geometry_sha256','opencv_binary_sha256','opencv_version','rules')}
    launch.update(source_sha256=sources,recorded_artifact_sha256=bound,scene_specification=spec,
        world_from_optical_poses=poses,recorded_pose_indices=[0,8],independent_builds=2,expected_frames=4,
        minimum_free_bytes=RESERVE,maximum_artifact_bytes=BUDGET,physics_steps=0,navigation_qualified=False)
    verify(launch)
    if len((json.dumps(launch,indent=2,allow_nan=False)+'\n').encode())>BUDGET//4:
        raise ValueError('serialized launch metadata exceeds reserved quarter of bench budget')
    if shutil.disk_usage(BASE.parent).free<RESERVE+BUDGET:raise ValueError('bench storage reserve required')
    return inv,launch


def capture(inv,launch,repeat):
    directory=OUTPUT/f'repeat_{repeat}';directory.mkdir();visual=directory/'visual_meshes';visual.mkdir()
    spec=launch['scene_specification'];build=None
    initialize_genesis(backend='cpu',seed=spec['physics_seed'],logging_level='warning')
    try:
        build=build_scene_from_pack(inv.pack(spec),output=visual,appearance_arm=spec['appearance_arm'],appearance_seed=spec['appearance_seed'])
        write_json(directory/'native_identity.json',dict(environment=build.native_environment_identity,robot=build.native_robot_geometry))
        for i,pose in enumerate(launch['world_from_optical_poses']):
            if shutil.disk_usage(BASE.parent).free<RESERVE:raise ValueError('bench free reserve exhausted')
            camera=build.camera;T=np.asarray(pose);check_capture_domain(build,T)
            camera.set_pose(pos=T[:3,3],lookat=T[:3,3]+T[:3,2],up=-T[:3,1]);check_optical_pose(camera.transform,T)
            np.testing.assert_allclose(camera.intrinsics,INTRINSICS,rtol=0,atol=1e-7)
            if camera.near!=.005 or camera.far!=200. or tuple(camera.res)!=(640,480):raise ValueError('unchanged5mm-near camera required')
            if camera._raytracer is not None or camera._batch_renderer is not None:raise ValueError('single native rasterizer required')
            before=np.array(camera.transform,copy=True)
            rgb=np.asarray(camera.render(rgb=True,depth=False,segmentation=False,normal=False)[0])
            depth=np.asarray(camera.render(rgb=False,depth=True,segmentation=False,normal=False)[1])
            if rgb.ndim==4 and rgb.shape[0]==1:rgb=rgb[0]
            if depth.ndim==3 and depth.shape[0]==1:depth=depth[0]
            rgb=rgb[...,:3]
            if int(build.scene.t)!=0 or not np.array_equal(before,camera.transform):raise ValueError('static same-epoch captures required')
            if rgb.shape!=(480,640,3) or rgb.dtype!=np.uint8 or depth.shape!=(480,640) or depth.dtype!=np.float32:
                raise ValueError('native array contract')
            sample=sampling_readback(camera)
            if sample!=dict(draw_framebuffer_is_single_sample_target=True,draw_framebuffer_is_multisample_target=False,
                    samples=0,sample_buffers=0,multisample_enabled=False,pixel_scale=1):raise ValueError('single-sample depth required')
            Image.fromarray(rgb).save(directory/f'rgb_{i:04d}.png')
            np.savez_compressed(directory/f'native_depth_{i:04d}.npz',optical_depth_m=depth)
            report=evaluate_visibility(depth,spec['geometry']['wall_boxes'],T,render_near_m=.005)
            write_json(directory/f'frame_{i:04d}.json',dict(repeat=repeat,pose_index=i,world_from_optical=pose,
                sampling_readback=sample,physical_visibility=report,physics_steps=0))
            print('UNION_RGB_CAPTURE',repeat,i,report,flush=True)
    finally:
        if build is not None:build.scene.destroy()
        shutdown_genesis()


def main():
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    inv,launch=preflight();create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    try:
        for repeat in (0,1):verify(launch);capture(inv,launch,repeat)
        names=[f'repeat_{r}/'+n for r in (0,1) for n in ['native_identity.json','visual_meshes/ground_visual.ply',
            'visual_meshes/wall_union_visual.ply']+[f'{prefix}_{i:04d}.{suffix}' for i in (0,1)
            for prefix,suffix in (('rgb','png'),('native_depth','npz'),('frame','json'))]]
        bindings={n:digest(OUTPUT/n) for n in names};verify_artifacts(OUTPUT,bindings);rows=[]
        for i,pose in enumerate(launch['world_from_optical_poses']):
            images=[];depths=[];reports=[]
            for r in (0,1):
                directory=OUTPUT/f'repeat_{r}'
                with Image.open(directory/f'rgb_{i:04d}.png') as image:images.append(np.array(image))
                depths.append(read_npz(directory,f'native_depth_{i:04d}.npz')['optical_depth_m'])
                report=evaluate_visibility(depths[-1],launch['scene_specification']['geometry']['wall_boxes'],pose,render_near_m=.005)
                if report!=read_json(directory,f'frame_{i:04d}.json')['physical_visibility']:raise ValueError('saved raw score mismatch')
                reports.append(report)
            rows.append(dict(pose_index=i,rgb_exact=bool(np.array_equal(*images)),depth_exact=bool(np.array_equal(*depths)),
                changed_rgb_pixels=int((images[0]!=images[1]).any(-1).sum()),physical_visibility=reports))
        used=(OUTPUT/'launch.json').stat().st_size+sum((OUTPUT/n).stat().st_size for n in names)
        if used>BUDGET:raise ValueError('fixed bench artifact budget exceeded')
        verify(launch);verify_artifacts(INPUT,launch['recorded_artifact_sha256']);verify_artifacts(OUTPUT,bindings)
        passed=all(r['rgb_exact'] and r['depth_exact'] and all(v['passes_sampled_physical_visibility'] for v in r['physical_visibility']) for r in rows)
        write_json(OUTPUT/'result.json',dict(status='UNION_WALL_NATIVE_REPEATABILITY_BENCH_COMPLETE',passes_fixed_bench=passed,
            comparisons=rows,artifact_sha256=bindings,artifact_bytes=used,physics_steps=0,frames=4,
            dynamic_repeatability_qualified=False,collection_replacement_authorized=False,navigation_qualified=False,goal_achieved=False))
        print('UNION_RGB_BENCH_COMPLETE',passed,rows,flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_UNION_RGB_BENCH_FAILURE',reason=repr(error)))
        raise


if __name__=='__main__':main()
