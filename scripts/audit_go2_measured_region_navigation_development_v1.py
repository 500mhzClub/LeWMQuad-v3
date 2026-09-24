#!/usr/bin/env python3
"""Full original mission replay plus live depth-state replay and motion metrics."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'lewm_genesis', ROOT/'lewm_worlds'): sys.path.insert(0, str(path))
from lewm.causal_depth_observation_development import INTRINSICS, from_native_depth, validate_depth, body_points
from lewm.depth_relative_motion_development import DepthRelativeState
from lewm.depth_motion_evaluation_development import moving_depth_check, reduce_motion
from lewm.visual_surface_depth_evaluation_development import check_floor_identity
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.audit_measured_region_rgb_core_development import audit_trial as audit_rgb_trial
from scripts.run_go2_whole_task_navigation_sampling_correction_development_v1 import LEAVES as RGB_LEAVES
from scripts.audit_go2_local_control_factorial_development_v1 import check, read_npz
from scripts.run_go2_measured_region_navigation_development_v1 import (
    OUTPUT, SCHEMA, LEAVES, trial_specs, verify_native, verify_bindings, digest, write_json)


def audit_trial(directory, spec, row):
    count = row['rgb_packets']
    leaves = {*LEAVES, *[name for i in range(count) for name in
                        (f'rgb_{i:04d}.png', f'depth_{i:04d}.npz', f'native_depth_{i:04d}.npz')]}
    check(set(row['artifact_sha256']) == leaves, 'exact moving RGBD artifact population')
    for name, sha in row['artifact_sha256'].items(): check(digest(directory/name) == sha, 'RGBD artifact identity')
    base_names = {*RGB_LEAVES, *(f'rgb_{i:04d}.png' for i in range(count))}
    rgb_result = audit_rgb_trial(directory, spec, {**row,
        'artifact_sha256': {n: row['artifact_sha256'][n] for n in base_names}})
    check(row['floor_identity'] == check_floor_identity(json.loads((directory/'floor_visual_collision_identity.json').read_text())),
          'actual fixed visual/collision floor identity')
    cameras = json.loads((directory/'camera_audit.json').read_text())
    depth_cameras = json.loads((directory/'depth_camera_audit.json').read_text())
    observations = json.loads((directory/'relative_state_observations.json').read_text())
    check(len(cameras) == len(depth_cameras) == len(observations) == len(row['depth_checks']) == count, 'complete RGBD and observer population')
    observer = DepthRelativeState()
    checks = []
    for index, camera in enumerate(cameras):
        policy, depth = load_rgbd_observation(directory, index)
        now = policy['sensor_state']['decision_ns']
        native = read_npz(directory/f'native_depth_{index:04d}.npz')
        check(set(native) == {'optical_depth_m'}, 'only native optical depth array')
        native = native['optical_depth_m']
        meta = depth_cameras[index]
        check(set(meta) == {'timestamp_s', 'physical_sample_index', 'native_shape', 'native_dtype', 'native_intrinsics',
            'native_near_m', 'native_far_m', 'native_vertical_fov_deg', 'native_depth_sha256', 'same_render_call_as_rgb',
            'renderer', 'representation', 'hardware_calibrated', 'same_physics_and_camera_as_rgb',
            'physics_clock_before_after_ns', 'sampling_readback'}, 'exact native camera evidence fields')
        check(meta['same_render_call_as_rgb'] is False and meta['same_physics_and_camera_as_rgb'] is True and meta['renderer'] == 'genesis_rasterizer'
              and meta['representation'] == 'optical_axis_depth_m' and meta['hardware_calibrated'] is False,
              'declared actual depth path')
        check(meta['timestamp_s'] == camera['timestamp_s'] and meta['physical_sample_index'] == camera['physical_sample_index']
              and meta['native_shape'] in ([480, 640], [1, 480, 640]) and meta['native_dtype'] == 'float32', 'actual depth clock/shape')
        check(meta['native_near_m'] == .05 and meta['native_far_m'] == 200.
              and np.allclose(meta['native_intrinsics'], INTRINSICS, atol=1e-7, rtol=0), 'native camera calibration')
        check(meta['physics_clock_before_after_ns'] == [now, now], 'no intervening physics')
        check(meta['sampling_readback'] == {'draw_framebuffer_is_single_sample_target': True,
            'draw_framebuffer_is_multisample_target': False, 'samples': 0, 'sample_buffers': 0,
            'multisample_enabled': False, 'pixel_scale': 1}, 'native depth single-sample readback')
        focal = 240/np.tan(np.deg2rad(meta['native_vertical_fov_deg']/2))
        check(abs(focal-INTRINSICS[0][0]) < 1e-7, 'native vertical field of view')
        check(meta['native_depth_sha256'] == hashlib.sha256(native.tobytes()).hexdigest(), 'actual depth identity')
        expected = from_native_depth(native, policy, measured_ns=now, available_ns=now, now_ns=now)
        for key in expected:
            check(np.array_equal(depth[key], expected[key]) if isinstance(expected[key], np.ndarray)
                  else depth[key] == expected[key], 'exact depth acquisition replay')
        validate_depth(depth, policy, now_ns=now)
        cloud = body_points(depth, policy, now_ns=now)
        check(np.isnan(cloud['points_body_m'][~cloud['valid']]).all(), 'unknown depth never made free')

        result = moving_depth_check(native, spec['geometry']['wall_boxes'], camera['world_from_optical'])
        check(result == row['depth_checks'][index], 'moving depth metric replay')
        checks.append(result)
        expected_state = ({'measured_ns': now, 'status': 'NON_DECISION_TERMINAL_CAPTURE'} if now % 100_000_000 else
                          observer.observe(policy, depth, load_fast_packet(directory, index), now_ns=now))
        expected_record = {'observation_index': index, 'observer': expected_state}
        check(json.loads(json.dumps(expected_record, allow_nan=False)) == observations[index], 'exact live sensor-only state replay')
    raw = read_npz(directory/'physics_trace.npz')
    motion = reduce_motion(raw, cameras, observations)
    check(motion == row['moving_state'], 'independent measured-versus-physical motion reduction')
    return {'scene_id': spec['scene_id'], 'rgb_audit': rgb_result, 'moving_state': motion,
            'all_moving_depth_checks_pass': all(r['passes'] for r in checks), 'depth_checks': checks}


def main():
    if len(sys.argv) != 1 or (OUTPUT/'raw_artifact_audit.json').exists():
        raise ValueError('fixed fresh moving RGBD audit required')
    launch = json.loads((OUTPUT/'launch.json').read_text())
    report = json.loads((OUTPUT/'result.json').read_text())
    check(launch['schema'] == SCHEMA, 'distinct moving RGBD study schema')
    check(report['status'] == 'COMPLETE' and report['completed_trials'] == report['planned_trials'] == 2
          and len(report['trials']) == 2 and launch['trial_specs'] == trial_specs(), 'complete fixed moving population')
    check(report['launch_sha256'] == digest(OUTPUT/'launch.json'), 'launch binding')
    verify_bindings(launch['source_sha256'] | launch['input_sha256'] | launch['gait_sha256'])
    check(launch['native_renderer_sha256'] == verify_native(), 'native source binding')
    rows = []
    try:
        for spec, row in zip(trial_specs(), report['trials'], strict=True):
            directory = OUTPUT/spec['scene_id']
            check(json.loads((directory/'result.json').read_text()) == row, 'member/root identity')
            rows.append(audit_trial(directory, spec, row))
            print(json.dumps({'event': 'audited', 'completed': len(rows),
                'moving_state_pass': rows[-1]['moving_state']['passes_declared_moving_state_check'],
                'moving_depth_pass': rows[-1]['all_moving_depth_checks_pass']}), flush=True)
        verify_bindings(launch['source_sha256'] | launch['input_sha256'] | launch['gait_sha256'])
        check(launch['native_renderer_sha256'] == verify_native(), 'terminal native binding')
        result = {'status': 'PASS', 'audited_trials': 2, 'trials': rows,
                  'study_result_sha256': digest(OUTPUT/'result.json'), 'audit_source_sha256': digest(Path(__file__)),
                  'scope': 'continuous measured-region development controller and physical outcomes; no independent-maze/JEPA/hardware qualification'}
        write_json(OUTPUT/'raw_artifact_audit.json', result)
        print(json.dumps({'status': 'PASS', 'audited_trials': 2}), flush=True)
    except Exception as error:
        write_json(OUTPUT/'raw_artifact_audit.json', {'status': 'FAIL', 'error': repr(error), 'trials': rows,
                                                    'study_result_sha256': digest(OUTPUT/'result.json')})
        raise


if __name__ == '__main__': main()
