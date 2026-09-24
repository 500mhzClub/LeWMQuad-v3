#!/usr/bin/env python3
"""Raw RGB/body audit plus independent native depth and sensor-clock checks."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'lewm_genesis', ROOT/'lewm_worlds'): sys.path.insert(0, str(path))
from lewm.causal_depth_observation_development import (INTRINSICS, CausalDepthHistory, from_native_depth,
                                                       body_points, validate_depth)
from lewm.depth_geometry_evaluation_development import evaluate_depth
from lewm.fast_gyro_development import validate_fast_packet
from lewm.physical_execution_development import rotation_xyzw
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.audit_go2_marker_beacon_development_v2 import audit_trial as audit_rgb_trial
from scripts.audit_go2_local_control_factorial_development_v1 import check, read_npz
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.run_go2_rgbd_observation_development_v1 import (OUTPUT, LEAVES, RGB_LEAVES, trials,
                                                            verify_native, verify_bindings, digest, write_json)


def audit_trial(directory, spec, row):
    leaves = {*LEAVES, *[name for i in range(5) for name in
                        (f'rgb_{i:04d}.png', f'depth_{i:04d}.npz', f'native_depth_{i:04d}.npz')]}
    check(set(row['artifact_sha256']) == leaves, 'exact RGBD artifact population')
    for name, sha in row['artifact_sha256'].items(): check(digest(directory/name) == sha, 'RGBD artifact binding')
    # The predecessor's complete raw RGB/body audit checks its exact own subset;
    # additional sensor artifacts are checked here, not hidden from accounting.
    base_names = {*RGB_LEAVES, *(f'rgb_{i:04d}.png' for i in range(5))}
    base = {**row, 'artifact_sha256': {n: row['artifact_sha256'][n] for n in base_names}}
    rgb_result = audit_rgb_trial(directory, spec, base)
    raw = read_npz(directory/'physics_trace.npz')
    fast = read_npz(directory/'fast_gyro_samples.npz')
    history = read_npz(directory/'fast_gyro_histories.npz')
    times = np.rint(raw['timestamp_s']*1e9).astype(np.int64)
    rates = np.stack([rotation_xyzw(p[3:]).T@v[3:] for p, v in zip(raw['base_pose_world'], raw['base_twist_world'], strict=True)])
    check(set(fast) == set(history) == {'values', 'valid', 'measured_ns', 'available_ns'}, 'fast sensor fields')
    check(fast['values'].shape == (950, 3) and fast['valid'].shape == (950, 3) and fast['valid'].dtype == bool,
          'native fast sensor shapes')
    check(fast['valid'].all() and np.allclose(fast['values'], rates, atol=1e-12, rtol=0)
          and np.array_equal(fast['measured_ns'], times) and np.array_equal(fast['available_ns'], times), 'fast sensor physics reconstruction')
    check(all(len(v) == 5 for v in history.values()), 'fast camera history population')
    cameras = json.loads((directory/'camera_audit.json').read_text())
    depth_cameras = json.loads((directory/'depth_camera_audit.json').read_text())
    check(len(depth_cameras) == 5 and len(row['depth_checks']) == 5, 'paired native depth population')
    stream = CausalDepthHistory(); checks = []
    for index, camera in enumerate(cameras):
        policy, depth = load_rgbd_observation(directory, index)
        now = policy['sensor_state']['decision_ns']
        fast_packet = load_fast_packet(directory, index)
        validate_fast_packet(fast_packet, policy, now_ns=now)
        selected = np.flatnonzero((times >= now-100_000_000) & (times <= now))[-51:]
        expected_fast = {'values': rates[selected], 'valid': np.ones((51, 3), bool),
                         'measured_ns': times[selected], 'available_ns': times[selected]}
        for field, value in expected_fast.items():
            check(np.array_equal(history[field][index], value), 'exact fast sensor camera history')
        native = read_npz(directory/f'native_depth_{index:04d}.npz')
        check(set(native) == {'optical_depth_m'}, 'only native optical depth array')
        native = native['optical_depth_m']
        meta = depth_cameras[index]
        check(set(meta) == {'timestamp_s', 'physical_sample_index', 'native_shape', 'native_dtype', 'native_intrinsics',
            'native_near_m', 'native_far_m', 'native_vertical_fov_deg', 'native_depth_sha256', 'same_render_call_as_rgb',
            'renderer', 'representation', 'hardware_calibrated'}, 'exact native camera evidence fields')
        check(meta['same_render_call_as_rgb'] is True and meta['renderer'] == 'genesis_rasterizer'
              and meta['representation'] == 'optical_axis_depth_m' and meta['hardware_calibrated'] is False,
              'declared actual depth path')
        check(meta['timestamp_s'] == camera['timestamp_s'] and meta['physical_sample_index'] == camera['physical_sample_index']
              and meta['native_shape'] in ([480, 640], [1, 480, 640]) and meta['native_dtype'] == 'float32', 'actual depth clock/shape')
        check(meta['native_near_m'] == .05 and meta['native_far_m'] == 200.
              and np.allclose(meta['native_intrinsics'], INTRINSICS, atol=1e-7, rtol=0), 'native camera calibration')
        focal = 240/np.tan(np.deg2rad(meta['native_vertical_fov_deg']/2))
        check(abs(focal-INTRINSICS[0][0]) < 1e-7, 'native vertical field of view')
        check(meta['native_depth_sha256'] == hashlib.sha256(native.tobytes()).hexdigest(), 'actual depth identity')
        expected = from_native_depth(native, policy, measured_ns=now, available_ns=now, now_ns=now)
        for key in expected:
            check(np.array_equal(depth[key], expected[key]) if isinstance(expected[key], np.ndarray)
                  else depth[key] == expected[key], 'exact depth acquisition replay')
        validate_depth(depth, policy, now_ns=now); stream.push(depth, policy, now_ns=now)
        cloud = body_points(depth, policy, now_ns=now)
        check(np.isnan(cloud['points_body_m'][~cloud['valid']]).all(), 'unknown depth never made free')
        result = evaluate_depth(native, spec['geometry']['wall_boxes'], camera['world_from_optical'], marker_case=spec['marker_case'])
        check(result == row['depth_checks'][index], 'independent ray-reference outcome replay')
        checks.append(result)
    return {'scene_id': spec['scene_id'], 'rgb_audit': rgb_result, 'depth_checks': checks,
            'depth_interface_pass': all(r['passes_declared_depth_check'] for r in checks),
            'physics_samples': 950, 'rgbd_packets': 5, 'depth_history_retained': len(stream.snapshot())}


def main():
    if len(sys.argv) != 1 or (OUTPUT/'raw_artifact_audit.json').exists():
        raise ValueError('fixed fresh RGBD audit required')
    launch = json.loads((OUTPUT/'launch.json').read_text())
    report = json.loads((OUTPUT/'result.json').read_text())
    check(report['status'] == 'COMPLETE' and report['completed_trials'] == report['planned_trials'] == 2
          and len(report['trials']) == 2 and launch['trial_specs'] == trials(), 'complete fixed RGBD population')
    check(report['launch_sha256'] == digest(OUTPUT/'launch.json'), 'RGBD launch identity')
    verify_bindings(launch['source_sha256'] | launch['input_sha256'] | launch['gait_sha256'])
    check(launch['native_renderer_sha256'] == verify_native(), 'reviewed renderer identity')
    rows = []
    try:
        for spec, row in zip(trials(), report['trials'], strict=True):
            directory = OUTPUT/spec['scene_id']
            check(json.loads((directory/'result.json').read_text()) == row, 'member/root identity')
            rows.append(audit_trial(directory, spec, row))
            print(json.dumps({'event': 'audited', 'completed': len(rows), 'depth_interface_pass': rows[-1]['depth_interface_pass']}), flush=True)
        verify_bindings(launch['source_sha256'] | launch['input_sha256'] | launch['gait_sha256'])
        check(launch['native_renderer_sha256'] == verify_native(), 'terminal renderer identity')
        result = {'status': 'PASS', 'audited_trials': 2, 'trials': rows,
                  'all_depth_checks_pass': all(r['depth_interface_pass'] for r in rows),
                  'study_result_sha256': digest(OUTPUT/'result.json'), 'audit_source_sha256': digest(Path(__file__)),
                  'scope': 'actual RGBD interface evidence, not navigation or hardware qualification'}
        write_json(OUTPUT/'raw_artifact_audit.json', result)
        print(json.dumps({k: result[k] for k in ('status', 'audited_trials', 'all_depth_checks_pass')}), flush=True)
    except Exception as error:
        write_json(OUTPUT/'raw_artifact_audit.json', {'status': 'FAIL', 'error': repr(error), 'trials': rows,
                                                    'study_result_sha256': digest(OUTPUT/'result.json')})
        raise


if __name__ == '__main__': main()
