"""Fixed-order causal raster bench: two orders, two independent builds each."""
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from lewm.causal_depth_observation_development import INTRINSICS
from lewm.physical_first_surface_depth_development import evaluate_visibility
from lewm_genesis.union_wall_rgbd_scene_development import build_scene_from_pack
from lewm_genesis.ordered_union_raster_development import ORDERS,install_order,verify_order
from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
from lewm_genesis.bounded_scene_builder_development import check_capture_domain
from lewm_genesis.optical_camera_readback_development import check_optical_pose
from scripts.single_sample_rgbd_session_development import sampling_readback
from scripts.independent_layout_batch_development import load_inventory
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json,read_npz
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.navigation_artifact_root_development import BASE,validate_root,create_output,verify_artifacts
from scripts.diagnose_go2_recorded_raster_failures_v1 import UNION,UNION_IDS

OUTPUT=BASE/'go2_ordered_union_rgb_repeatability_probe_v1_attempt_001'
PROTOCOL='docs/go2_ordered_union_rgb_repeatability_probe_v1_2026-09-06.md'
SOURCE='scripts/probe_go2_ordered_union_rgb_repeatability_v1.py'
RESERVE=40*1024**3
BUDGET=256*1024**2


def verify_ordered_launch(launch):
    verify(launch)
    path=Path(next(p for p in launch['native_sha256'] if p.endswith('/ext/pyrender/renderer.py'))).with_name('scene.py')
    if launch['native_scene_sha256']!={str(path):digest(path)}:raise ValueError('native sorting implementation changed')


def preflight():
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive distinct bench; no retry/resume')
    verify_artifacts(UNION,UNION_IDS);old=read_json(UNION,'launch.json');verify(old)
    result=read_json(UNION,'result.json');verify_artifacts(UNION,result['artifact_sha256'])
    inv=load_inventory()
    launch={k:old[k] for k in ('input_sha256','native_sha256','native_geometry_sha256','opencv_binary_sha256','opencv_version','rules')}
    native_scene=Path(next(p for p in launch['native_sha256'] if p.endswith('/ext/pyrender/renderer.py'))).with_name('scene.py')
    launch['native_scene_sha256']={str(native_scene):digest(native_scene)}
    sources=discover_sources((PROTOCOL,SOURCE,'lewm/tests/test_ordered_union_raster_development.py'),old['source_sha256'])
    launch.update(source_sha256=sources,predecessor_sha256=UNION_IDS|result['artifact_sha256'],
        scene_specification=old['scene_specification'],world_from_optical_poses=old['world_from_optical_poses'],
        orders=list(ORDERS),independent_builds_per_order=2,expected_frames=8,physics_steps=0,
        minimum_free_bytes=RESERVE,maximum_artifact_bytes=BUDGET,navigation_qualified=False)
    assert inv.specification(launch['scene_specification']['trial'])==launch['scene_specification']
    verify_ordered_launch(launch)
    if len((json.dumps(launch,indent=2,allow_nan=False)+'\n').encode())>BUDGET//4:raise ValueError('metadata budget')
    if shutil.disk_usage(BASE.parent).free<RESERVE+BUDGET:raise ValueError('storage reserve')
    return inv,launch


def capture(inv,launch,repeat,order):
    directory=OUTPUT/f'repeat_{repeat}_{order}';directory.mkdir();visual=directory/'visual_meshes';visual.mkdir()
    spec=launch['scene_specification'];build=None
    initialize_genesis(backend='cpu',seed=spec['physics_seed'],logging_level='warning')
    try:
        build=build_scene_from_pack(inv.pack(spec),output=visual,appearance_arm=spec['appearance_arm'],appearance_seed=spec['appearance_seed'])
        witness=install_order(build.camera._rasterizer._context._scene,order)
        write_json(directory/'raster_order.json',witness)
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
            verify_order(camera,witness)
            sample=sampling_readback(camera)
            if sample!=dict(draw_framebuffer_is_single_sample_target=True,draw_framebuffer_is_multisample_target=False,
                    samples=0,sample_buffers=0,multisample_enabled=False,pixel_scale=1):raise ValueError('single-sample depth required')
            Image.fromarray(rgb).save(directory/f'rgb_{i:04d}.png')
            np.savez_compressed(directory/f'native_depth_{i:04d}.npz',optical_depth_m=depth)
            report=evaluate_visibility(depth,spec['geometry']['wall_boxes'],T,render_near_m=.005)
            write_json(directory/f'frame_{i:04d}.json',dict(repeat=repeat,pose_index=i,world_from_optical=pose,
                sampling_readback=sample,physical_visibility=report,physics_steps=0))
            print('ORDERED_RGB_CAPTURE',order,repeat,i,report,flush=True)
    finally:
        if build is not None:build.scene.destroy()
        shutdown_genesis()


def main():
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    inv,launch=preflight();create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    try:
        for order in ORDERS:
            for repeat in (0,1):verify_ordered_launch(launch);capture(inv,launch,repeat,order)
        names=[f'repeat_{r}_{o}/'+n for o in ORDERS for r in (0,1)
            for n in ['native_identity.json','raster_order.json','visual_meshes/ground_visual.ply','visual_meshes/wall_union_visual.ply']
            +[f'{p}_{i:04d}.{s}' for i in (0,1) for p,s in (('rgb','png'),('native_depth','npz'),('frame','json'))]]
        bindings={n:digest(OUTPUT/n) for n in names};verify_artifacts(OUTPUT,bindings)
        rows=[]
        for order in ORDERS:
            for i in (0,1):
                dirs=[OUTPUT/f'repeat_{r}_{order}' for r in (0,1)]
                rgb=[np.asarray(Image.open(d/f'rgb_{i:04d}.png')) for d in dirs]
                depths=[read_npz(d,f'native_depth_{i:04d}.npz')['optical_depth_m'] for d in dirs]
                reports=[evaluate_visibility(n,launch['scene_specification']['geometry']['wall_boxes'],
                    launch['world_from_optical_poses'][i],render_near_m=.005) for n in depths]
                assert all(report==read_json(d,f'frame_{i:04d}.json')['physical_visibility'] for report,d in zip(reports,dirs))
                rows.append(dict(order=order,pose_index=i,rgb_exact=bool(np.array_equal(*rgb)),
                    depth_exact=bool(np.array_equal(*depths)),changed_rgb_pixels=int(np.any(rgb[0]!=rgb[1],axis=2).sum()),
                    physical_visibility=reports))
        across=[]
        for r in (0,1):
            for i in (0,1):
                rgb=[np.array(Image.open(OUTPUT/f'repeat_{r}_{o}/rgb_{i:04d}.png')) for o in ORDERS]
                across.append(dict(repeat=r,pose_index=i,changed_rgb_pixels=int(np.any(rgb[0]!=rgb[1],axis=2).sum())))
        verify_ordered_launch(launch);verify_artifacts(UNION,launch['predecessor_sha256']);verify_artifacts(OUTPUT,bindings)
        used=(OUTPUT/'launch.json').stat().st_size+sum((OUTPUT/n).stat().st_size for n in names)
        if used>BUDGET or shutil.disk_usage(BASE.parent).free<RESERVE:raise ValueError('terminal storage budget')
        result=dict(status='ORDERED_UNION_NATIVE_REPEATABILITY_BENCH_COMPLETE',
            passes_fixed_bench=all(r['rgb_exact'] and r['depth_exact'] and all(p['passes_sampled_physical_visibility']
                for p in r['physical_visibility']) for r in rows),
            comparisons=rows,between_order_comparisons=across,artifact_sha256=bindings,artifact_bytes=used,
            frames=8,physics_steps=0,dynamic_repeatability_qualified=False,collection_replacement_authorized=False,
            navigation_qualified=False,goal_achieved=False)
        write_json(OUTPUT/'result.json',result)
        print('ORDERED_RGB_BENCH_COMPLETE',result['passes_fixed_bench'],rows,across,flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='ORDERED_UNION_NATIVE_BENCH_FAILURE',reason=repr(error)))
        raise


if __name__=='__main__':main()
