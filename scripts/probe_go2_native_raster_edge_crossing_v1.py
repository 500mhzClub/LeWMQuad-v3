"""Nine fixed native views across a foreground silhouette; zero physics."""
import json
import shutil
import cv2
import numpy as np
import torch
from PIL import Image
from lewm.raster_edge_crossing_development import poses, score_frame
from lewm.causal_depth_observation_development import INTRINSICS
from lewm_genesis.union_wall_rgbd_scene_development import build_scene_from_pack
from lewm_genesis.ordered_union_raster_development import install_order, verify_order
from lewm_genesis.core_raster_precision_development import precision_readback
from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
from lewm_genesis.bounded_scene_builder_development import check_capture_domain
from lewm_genesis.optical_camera_readback_development import check_optical_pose
from scripts.single_sample_rgbd_session_development import sampling_readback
from scripts.independent_layout_batch_development import load_inventory
from scripts.check_go2_terminal_event_coverage_v1 import INPUT, IDENTITIES
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch
from scripts.startup_source_inventory_development import discover_sources
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json, read_npz
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

OUTPUT = BASE / 'go2_native_raster_edge_crossing_v1_attempt_001'
PROTOCOL = 'docs/go2_native_raster_edge_crossing_v1_2026-09-06.md'
RUN = 'repeat_0_l00_junction_recent_forward_nominal_a3'
TRIAL = 'l00_junction_recent_forward_nominal_a3'
BUDGET = 128 * 1024**2
RESERVE = 40 * 1024**3


def preflight():
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive edge assay; no retry/resume')
    verify_artifacts(INPUT, IDENTITIES)
    old = read_json(INPUT, 'launch.json'); verify_ordered_launch(old)
    result = read_json(INPUT, 'result.json')
    commit_name = 'episode_001_commit.json'
    bound = IDENTITIES | {commit_name: result['commits'][commit_name]}
    verify_artifacts(INPUT, bound)
    commit = read_json(INPUT, commit_name)
    if commit['run'] != RUN or commit['trial'] != TRIAL: raise ValueError('exact fixed scene identity')
    for name in ('specification.json', 'camera_audit.json', 'native_depth_0011.npz'):
        bound[RUN + '/' + name] = commit['artifact_sha256'][RUN + '/' + name]
    verify_artifacts(INPUT, bound)
    inv = load_inventory(); spec = inv.specification(TRIAL)
    if read_json(INPUT / RUN, 'specification.json') != spec: raise ValueError('inventory scene mismatch')
    base = read_json(INPUT / RUN, 'camera_audit.json')[11]['world_from_optical']
    sources = discover_sources((PROTOCOL, 'scripts/probe_go2_native_raster_edge_crossing_v1.py',
        'lewm/tests/test_raster_edge_crossing_development.py'), old['source_sha256'])
    launch = {k: old[k] for k in ('input_sha256', 'native_sha256', 'native_scene_sha256',
        'native_geometry_sha256', 'opencv_binary_sha256', 'opencv_version', 'rules')}
    launch.update(source_sha256=sources, recorded_artifact_sha256=bound, output_root=str(OUTPUT),
        scene_specification=spec, poses=poses(base), expected_frames=9, physics_steps=0,
        maximum_artifact_bytes=BUDGET, minimum_free_bytes=RESERVE, navigation_qualified=False)
    verify_ordered_launch(launch)
    if len((json.dumps(launch, indent=2, allow_nan=False) + '\n').encode()) > BUDGET // 4:
        raise ValueError('serialized launch allowance')
    if shutil.disk_usage(BASE).free < RESERVE + BUDGET: raise ValueError('edge assay storage reserve')
    return inv, launch


def capture(inv, launch):
    spec = launch['scene_specification']; build = None
    visual = OUTPUT / 'visual_meshes'; visual.mkdir()
    initialize_genesis(backend='cpu', seed=spec['physics_seed'], logging_level='warning')
    try:
        build = build_scene_from_pack(inv.pack(spec), output=visual,
            appearance_arm=spec['appearance_arm'], appearance_seed=spec['appearance_seed'])
        witness = install_order(build.camera._rasterizer._context._scene, 'floor_first')
        write_json(OUTPUT / 'native_identity.json', dict(environment=build.native_environment_identity,
            robot=build.native_robot_geometry, order=witness))
        for i, row in enumerate(launch['poses']):
            verify_ordered_launch(launch)
            if shutil.disk_usage(BASE).free < RESERVE: raise ValueError('edge assay reserve')
            T = np.asarray(row['world_from_optical']); camera = build.camera
            check_capture_domain(build, T)
            camera.set_pose(pos=T[:3, 3], lookat=T[:3, 3] + T[:3, 2], up=-T[:3, 1])
            check_optical_pose(camera.transform, T)
            np.testing.assert_allclose(camera.intrinsics, INTRINSICS, atol=1e-7, rtol=0)
            if camera.near != .005 or camera.far != 200. or tuple(camera.res) != (640, 480):
                raise ValueError('fixed native camera required')
            if camera._raytracer is not None or camera._batch_renderer is not None:
                raise ValueError('single-camera rasterizer required')
            before = np.array(camera.transform, copy=True)
            rgb = np.asarray(camera.render(rgb=True, depth=False, segmentation=False, normal=False)[0])
            depth = np.asarray(camera.render(rgb=False, depth=True, segmentation=False, normal=False)[1])
            if rgb.ndim == 4 and rgb.shape[0] == 1: rgb = rgb[0]
            if depth.ndim == 3 and depth.shape[0] == 1: depth = depth[0]
            rgb = rgb[..., :3]
            if int(build.scene.t) != 0 or not np.array_equal(before, camera.transform):
                raise ValueError('zero physics and unchanged capture epoch required')
            if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8 or depth.shape != (480, 640) or depth.dtype != np.float32:
                raise ValueError('native array contract')
            verify_order(camera, witness)
            sampling = sampling_readback(camera)
            if sampling != dict(draw_framebuffer_is_single_sample_target=True, draw_framebuffer_is_multisample_target=False,
                samples=0, sample_buffers=0, multisample_enabled=False, pixel_scale=1):
                raise ValueError('single-sample native depth required')
            precision = precision_readback(camera)
            Image.fromarray(rgb).save(OUTPUT / f'rgb_{i:04d}.png')
            np.savez_compressed(OUTPUT / f'native_depth_{i:04d}.npz', optical_depth_m=depth)
            score = score_frame(depth, spec['geometry']['wall_boxes'], T)
            write_json(OUTPUT / f'frame_{i:04d}.json', row | dict(score=score, precision=precision,
                sampling=sampling, physics_steps=0))
            print('EDGE_FRAME', i, row['principal_axis_offset_pixels'], score, flush=True)
    finally:
        if build is not None: build.scene.destroy()
        shutdown_genesis()


def main():
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    inv, launch = preflight(); create_output(OUTPUT); write_json(OUTPUT / 'launch.json', launch)
    try:
        capture(inv, launch)
        names = ['native_identity.json', 'visual_meshes/ground_visual.ply', 'visual_meshes/wall_union_visual.ply']
        names += [f'{p}_{i:04d}.{s}' for i in range(9) for p, s in (('rgb', 'png'), ('native_depth', 'npz'), ('frame', 'json'))]
        bindings = {n: digest(OUTPUT / n) for n in names}
        rows = []
        for i, pose in enumerate(launch['poses']):
            native = read_npz(OUTPUT, f'native_depth_{i:04d}.npz')['optical_depth_m']
            score = score_frame(native, launch['scene_specification']['geometry']['wall_boxes'], pose['world_from_optical'])
            if score != read_json(OUTPUT, f'frame_{i:04d}.json')['score']: raise ValueError('saved score mismatch')
            rows.append(score)
        baseline = np.array_equal(read_npz(OUTPUT, 'native_depth_0004.npz')['optical_depth_m'],
            read_npz(INPUT / RUN, 'native_depth_0011.npz')['optical_depth_m'])
        passed = bool(baseline and all(r['footprint']['stable_interior_metric_pass']
            and not r['footprint']['near_occlusion_failure'] and r['interior_negative_control']['rejected'] for r in rows))
        verify_ordered_launch(launch); verify_artifacts(INPUT, launch['recorded_artifact_sha256']); verify_artifacts(OUTPUT, bindings)
        result = dict(status='NATIVE_EDGE_CROSSING_ASSAY_COMPLETE', accounting_mechanism_pass=passed,
            baseline_depth_exact=bool(baseline), frames=9, physics_steps=0, scores=rows, artifact_sha256=bindings,
            boundary_pixels_certified=False, policy_free_space_certified=False, navigation_qualified=False, goal_achieved=False)
        used = (OUTPUT / 'launch.json').stat().st_size + sum((OUTPUT / n).stat().st_size for n in names)
        if used + len((json.dumps(result, indent=2, allow_nan=False) + '\n').encode()) > BUDGET:
            raise ValueError('edge assay total artifact budget')
        write_json(OUTPUT / 'result.json', result)
        print('EDGE_ASSAY_COMPLETE', passed, 'BASELINE_DEPTH_EXACT', bool(baseline), flush=True)
    except Exception as error:
        write_json(OUTPUT / 'failure.json', dict(status='TERMINAL_EDGE_ASSAY_FAILURE', reason=repr(error), goal_achieved=False))
        raise


if __name__ == '__main__': main()
