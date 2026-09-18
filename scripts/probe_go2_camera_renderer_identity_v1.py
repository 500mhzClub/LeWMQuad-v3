"""Read identity from an actual Genesis camera; no robot or physics steps."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from lewm_genesis.camera_renderer_identity_development import renderer_identity_readback
from lewm_genesis.core_raster_precision_development import precision_readback
from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
from scripts.single_sample_rgbd_session_development import sampling_readback
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT=BASE/'go2_camera_renderer_identity_v1_attempt_001'
PRIOR=BASE/'go2_live_maze_renderer_provenance_v1_attempt_001'
PRIOR_RESULT='76a592ec2bfa31927acb07cfdc6bb056de78df1693ab4066477f20a15159150a'
PRIOR_LAUNCH='bcef998cccc5acccf0fe0c7fe4537fe3b4aee6210b6438719f35826de1048cac'
PROTOCOL='docs/go2_camera_renderer_identity_v1_2026-09-09.md'
NATIVE=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/lib/python3.12/site-packages/genesis')
NATIVE_FILES=('vis/camera.py','vis/rasterizer.py','ext/pyrender/offscreen.py','ext/pyrender/renderer.py')
BUDGET=64*1024**2
SCENE=dict(physics_seed=2026090901,physics_dt_s=.002,plane_z_m=0.,
    fixed_box_position_m=[2.,0.,.5],fixed_box_size_m=[.3,1.,1.],
    camera_resolution=[640,480],camera_position_m=[0.,0.,.7],
    camera_lookat_m=[2.,0.,.5],camera_vertical_fov_deg=60.,near_m=.005,far_m=200.)


def verify_all(launch):
    verify(launch)
    verify_artifacts(PRIOR,{'result.json':PRIOR_RESULT,'launch.json':PRIOR_LAUNCH})
    for path,h in launch['additional_runtime_sha256'].items():
        if digest(Path(path))!=h:raise ValueError('installed camera/runtime source changed: '+path)


def capture():
    scene=None
    initialize_genesis(backend='cpu',seed=SCENE['physics_seed'],logging_level='warning')
    try:
        import genesis as gs
        scene=gs.Scene(sim_options=gs.options.SimOptions(dt=SCENE['physics_dt_s']),
            show_viewer=False,renderer=gs.renderers.Rasterizer())
        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(gs.morphs.Box(pos=SCENE['fixed_box_position_m'],size=SCENE['fixed_box_size_m'],fixed=True))
        camera=scene.add_camera(res=tuple(SCENE['camera_resolution']),pos=SCENE['camera_position_m'],
            lookat=SCENE['camera_lookat_m'],fov=SCENE['camera_vertical_fov_deg'],
            near=SCENE['near_m'],far=SCENE['far_m'],GUI=False)
        scene.build(n_envs=1)
        if camera._raytracer is not None or camera._batch_renderer is not None:
            raise ValueError('actual offscreen rasterizer camera required')
        pose=np.array(camera.transform,copy=True);records=[]
        for i in range(3):
            rgb=np.asarray(camera.render(rgb=True,depth=False,segmentation=False,normal=False)[0])
            depth=np.asarray(camera.render(rgb=False,depth=True,segmentation=False,normal=False)[1])
            if rgb.ndim==4 and rgb.shape[0]==1:rgb=rgb[0]
            if depth.ndim==3 and depth.shape[0]==1:depth=depth[0]
            rgb=np.array(rgb[...,:3],copy=True);depth=np.array(depth,copy=True)
            if (rgb.shape!=(480,640,3) or rgb.dtype!=np.uint8 or depth.shape!=(480,640)
                    or depth.dtype!=np.float32 or not np.isfinite(depth).all() or not np.any(depth<5.)):
                raise ValueError('nonempty fixed-resolution native RGB/depth required')
            identity=renderer_identity_readback(camera)
            sampling=sampling_readback(camera);precision=precision_readback(camera)
            if int(scene.t)!=0 or not np.array_equal(pose,camera.transform):
                raise ValueError('unchanged camera pose and zero physics required')
            filename=f'capture_{i:02d}.npz'
            np.savez_compressed(OUTPUT/filename,rgb=rgb,optical_depth_m=depth)
            records.append(dict(frame=i,identity=identity,sampling=sampling,precision=precision,
                rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest(),artifact=filename,
                camera_transform=np.asarray(camera.transform).tolist(),physics_steps=0))
            print('CAMERA_IDENTITY_CAPTURE',i,identity['renderer'],flush=True)
        for r in records[1:]:
            for key in ('identity','sampling','precision','rgb_sha256','depth_sha256','camera_transform'):
                if r[key]!=records[0][key]:raise ValueError('repeat rendering/readback changed: '+key)
        return records
    finally:
        try:
            if scene is not None:scene.destroy()
        finally:shutdown_genesis()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    if not __debug__:raise ValueError('assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive camera integration probe required')
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    verify_artifacts(PRIOR,{'result.json':PRIOR_RESULT,'launch.json':PRIOR_LAUNCH})
    old=read_json(PRIOR,'launch.json');verify(old)
    sources=discover_sources((PROTOCOL,'scripts/probe_go2_camera_renderer_identity_v1.py',
        'lewm/tests/test_camera_renderer_identity_development.py'),old['source_sha256'])
    runtime=old['additional_runtime_sha256']|{str(NATIVE/n):digest(NATIVE/n) for n in NATIVE_FILES}
    resources=hardware()
    launch={k:old[k] for k in ('input_sha256','native_sha256','native_scene_sha256',
        'native_geometry_sha256','opencv_binary_sha256','opencv_version','rules')}
    launch.update(protocol=PROTOCOL,output_root=str(OUTPUT),source_sha256=sources,
        additional_runtime_sha256=runtime,prior_result_sha256=PRIOR_RESULT,
        hardware=resources,scene_specification=SCENE,scenes=1,physics_steps=0,robot_entities=0,
        rgb_render_calls=3,depth_render_calls=3,camera_identity_readbacks=3,
        cpu_processes=1,numerical_threads=1,minimum_available_ram_bytes=16*1024**3,
        output_allowance_bytes=BUDGET,minimum_free_bytes=40*1024**3,os_resource_limits_enforced=False,
        render_environment={k:os.environ.get(k) for k in ('PYOPENGL_PLATFORM','LIBGL_ALWAYS_SOFTWARE',
            'EGL_DEVICE_ID','MESA_GL_VERSION_OVERRIDE','MESA_LOADER_DRIVER_OVERRIDE')},
        native_execution=True,model_loaded=False,model_training=False,robot_motion=False,
        concurrency_reason='one zero-step camera scene after paired timing finishes, beside one CPU settling replay',
        historical_camera_identity_inferred=False,navigation_qualified=False)
    verify_all(launch)
    memory_ok=resources['memory_available_bytes']>=16*1024**3
    storage_ok=resources['artifact_free_bytes']>=40*1024**3+BUDGET
    if args.preflight_only:
        print('CAMERA_IDENTITY_PREFLIGHT',json.dumps(dict(source_count=len(sources),hardware=resources,
            memory_admission_pass=memory_ok,storage_admission_pass=storage_ok,output_created=False)),flush=True);return
    if not memory_ok or not storage_ok:raise ValueError('bounded camera integration resources unavailable')
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('CAMERA_IDENTITY_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True);start=time.perf_counter()
    try:
        records=capture();verify_all(launch)
        artifacts={n:digest(OUTPUT/n) for n in ('launch.json',*(r['artifact'] for r in records))}
        verify_artifacts(OUTPUT,artifacts)
        if sum((OUTPUT/n).stat().st_size for n in artifacts)>BUDGET//2:
            raise ValueError('camera integration artifact headroom exhausted')
        write_json(OUTPUT/'result.json',dict(status='CAMERA_RENDERER_IDENTITY_INTEGRATION_COMPLETE',
            source_sha256=sources,artifact_sha256=artifacts,records=records,wall_s=time.perf_counter()-start,
            hardware_after=hardware(),actual_genesis_camera_context_queried=True,
            repeated_rgb_depth_after_readback_exact=True,scenes=1,physics_steps=0,robot_entities=0,
            historical_camera_identity_inferred=False,maze_renderer_equivalence_established=False,
            renderer_arithmetic_error_bound_proven=False,old_visibility_outcomes_unchanged=True,
            native_execution=True,model_training=False,navigation_qualified=False,goal_achieved=False))
        print('CAMERA_IDENTITY_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(reason=repr(error)));raise


if __name__=='__main__':main()
