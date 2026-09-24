"""One static20capture native regression for the unchanged mixed-height task."""
import json
import shutil
import cv2
import numpy as np
from PIL import Image
from lewm.geometry_progress_near_field_development import TRIALS,GEOMETRIES,APPEARANCES,assignments,specification,pack
from lewm.raster_footprint_visibility_development import evaluate_footprint
from lewm_genesis.variable_height_union_rgbd_scene_development import build_scene_from_pack
from lewm_genesis.ordered_union_raster_development import install_order,verify_order
from lewm_genesis.core_raster_precision_development import precision_readback
from lewm_genesis.optical_camera_readback_development import check_optical_pose
from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
from scripts.single_sample_rgbd_session_development import sampling_readback
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts

OUTPUT=BASE/'go2_geometry_progress_variable_height_render_bench_v1_attempt_001'
FAILED=BASE/'go2_geometry_progress_near_field_v1_attempt_001'
FAILED_IDS={'launch.json':'4fd1caf21cfa7cf5417693fab6607fe4b44a4404442cc3e81488b80494e4ae42',
    'failure.json':'f851d7496765993925a9f04dad1c3a528aa8dd983cdea2f0ab7fe609250958ab'}
ORIGINAL=BASE/'go2_geometry_progress_pilot_v1_attempt_001'
ORIGINAL_IDS={'launch.json':'265357dc47e8ceaecf7932f63441cd7020312c14bcce182e0cd07563888a2323',
    'result.json':'067a03f208cfc389e08ebba25b2c7a739dfb1b62f2001e1dc4238140cadfe6cc'}
PROTOCOL='docs/go2_geometry_progress_variable_height_render_bench_v1_2026-09-07.md'
FRAME_IDS=(0,15,16,17,18)
CASES=tuple(next(c for c,x in assignments().items() if x==dict(geometry=g,appearance_seed=s,action='forward'))
    for g in GEOMETRIES for s in APPEARANCES)


def main():
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists():raise ValueError('exclusive static variable-height bench')
    cv2.setNumThreads(1);verify_artifacts(FAILED,FAILED_IDS);verify_artifacts(ORIGINAL,ORIGINAL_IDS)
    old=read_json(FAILED,'launch.json');verify(old)
    failure=read_json(FAILED,'failure.json')
    if failure['completed_conditions'] or 'one common wall base and top required' not in failure['reason']:
        raise ValueError('exact pre-scene constructor failure required')
    original=read_json(ORIGINAL,'result.json')
    names=[c+'/'+n for c in CASES for n in ('specification.json','camera_audit.json')]
    inputs=ORIGINAL_IDS|{n:original['artifact_sha256'][n] for n in names};verify_artifacts(ORIGINAL,inputs)
    sources=discover_sources((PROTOCOL,'scripts/probe_go2_geometry_progress_variable_height_render_v1.py',
        'lewm/tests/test_variable_height_union_surface_development.py'),old['source_sha256'])
    definition=old|dict(source_sha256=sources);verify(definition)
    if shutil.disk_usage(BASE.parent).free<40*1024**3+256*1024**2:raise ValueError('bench storage reserve')
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',definition|dict(output_root=str(OUTPUT),
        failed_constructor_sha256=FAILED_IDS,source_pose_sha256=inputs,protocol=PROTOCOL,
        cases=list(CASES),frames=list(FRAME_IDS),physics_steps=0,maximum_new_bytes=256*1024**2))
    rows=[];products=['launch.json']
    try:
        for c in CASES:
            directory=OUTPUT/c;directory.mkdir();visual=directory/'visual_meshes';visual.mkdir()
            spec=specification(c);poses=read_json(ORIGINAL/c,'camera_audit.json');build=None
            initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
            try:
                build=build_scene_from_pack(pack(spec),output=visual,appearance_arm=spec['appearance_arm'],appearance_seed=spec['appearance_seed'])
                order=install_order(build.camera._rasterizer._context._scene,'floor_first')
                write_json(directory/'specification.json',spec)
                write_json(directory/'native_identity.json',dict(environment=build.native_environment_identity,robot=build.native_robot_geometry))
                products += [c+'/'+n for n in ('specification.json','native_identity.json','visual_meshes/ground_visual.ply','visual_meshes/wall_union_visual.ply')]
                for i in FRAME_IDS:
                    camera=build.camera;T=np.asarray(poses[i]['world_from_optical'])
                    camera.set_pose(pos=T[:3,3],lookat=T[:3,3]+T[:3,2],up=-T[:3,1]);check_optical_pose(camera.transform,T)
                    if camera.near!=.005 or camera.far!=200. or int(build.scene.t)!=0:raise ValueError('static5mm native camera required')
                    before=np.array(camera.transform,copy=True)
                    rgb=np.asarray(camera.render(rgb=True,depth=False,segmentation=False,normal=False)[0])
                    native=np.asarray(camera.render(rgb=False,depth=True,segmentation=False,normal=False)[1])
                    if rgb.ndim==4:rgb=rgb[0]
                    if native.ndim==3:native=native[0]
                    rgb=rgb[...,:3]
                    if (rgb.shape!=(480,640,3) or rgb.dtype!=np.uint8 or native.shape!=(480,640) or native.dtype!=np.float32
                            or int(build.scene.t)!=0 or not np.array_equal(before,camera.transform)):
                        raise ValueError('same-state native RGB/depth arrays required')
                    raster=dict(order=verify_order(camera,order),precision=precision_readback(camera),sampling=sampling_readback(camera))
                    score=evaluate_footprint(native,spec['geometry']['wall_boxes'],T,render_near_m=.005)
                    row=dict(trial=c,source_frame=i,source_physics_sample=poses[i]['physical_sample_index'],
                        world_from_optical=T.tolist(),physics_steps=0,raster=raster,score=score)
                    Image.fromarray(rgb).save(directory/f'rgb_{i:04d}.png')
                    np.savez_compressed(directory/f'native_depth_{i:04d}.npz',optical_depth_m=native)
                    write_json(directory/f'frame_{i:04d}.json',row);rows.append(row)
                    products += [c+'/'+n for n in (f'rgb_{i:04d}.png',f'native_depth_{i:04d}.npz',f'frame_{i:04d}.json')]
                    print('VARIABLE_HEIGHT_RENDER',c,i,score['stable_interior_metric_pass'],score['near_occlusion_failure'],flush=True)
            finally:
                if build is not None:build.scene.destroy()
                shutdown_genesis()
        verify(definition);verify_artifacts(ORIGINAL,inputs);verify_artifacts(FAILED,FAILED_IDS)
        used=sum((OUTPUT/n).stat().st_size for n in products)
        if used>256*1024**2:raise ValueError('static bench storage budget exceeded')
        write_json(OUTPUT/'result.json',dict(status='VARIABLE_HEIGHT_NATIVE_RENDER_BENCH_COMPLETE',
            frames=len(rows),cases=len(CASES),physics_steps=0,passed=bool(len(rows)==20 and all(
                r['score']['stable_interior_metric_pass'] and not r['score']['near_occlusion_failure'] for r in rows)),
            strict_failed_frames=[dict(trial=r['trial'],frame=r['source_frame']) for r in rows if not r['score']['original_strict_score']['passes_sampled_physical_visibility']],
            artifact_sha256={n:digest(OUTPUT/n) for n in products},source_sha256=sources,
            new_artifact_bytes=used,model_trained=False,navigation_qualified=False,goal_achieved=False))
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_VARIABLE_HEIGHT_RENDER_BENCH_FAILURE',reason=repr(error),completed_frames=len(rows)));raise


if __name__=='__main__':main()
