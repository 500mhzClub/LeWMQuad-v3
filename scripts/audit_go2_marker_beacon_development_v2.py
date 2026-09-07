#!/usr/bin/env python3
"""Correct only the native seven-value box encoding; preserve the V1 audit failure."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'lewm_genesis', ROOT/'lewm_worlds'):
    sys.path.insert(0, str(path))

from lewm.continuation_rgb_dataset_development import load_continuation_observation
from lewm.ground_projection_envelope_development import ORIGIN_BODY
from lewm.marker_beacon_scene_development import trials
from lewm.physical_execution_development import rotation_xyzw
from lewm.rgb_marker_beacon_development import MarkerDiscovery
from lewm.simulated_body_observation_development import SCHEMAS
from scripts.audit_go2_local_control_factorial_development_v1 import check, read_npz, recompute_contact_flags
from scripts.audit_go2_causal_rgb_body_capture_development_v1 import reconstruct_sensors, expected_history
from scripts.run_go2_marker_beacon_development_v1 import OUTPUT, LEAVES, digest, write_json, verify_bindings


def check_static_objects(spec, rows):
    boxes = spec['geometry']['wall_boxes']
    check(len(rows) == len(boxes), 'exact physical static-object population')
    palette = {'NEUTRAL_WALL': [.35, .35, .35], 'landmark_red': [.85, .12, .08],
               'landmark_blue': [.10, .22, .85]}
    for box, row in zip(boxes, rows, strict=True):
        expected = {'object_id': box['wall_id'], 'kind': 'wall', 'center_xyz_m': box['centre_xyz'],
                    'size_xyz_m': box['size_xyz'], 'yaw_rad': 0., 'material_id': box['material_id'],
                    'roll_rad': 0., 'pitch_rad': 0.}
        check(row['pack_object'] == expected and row['native_name'] == box['wall_id'], 'static name/pack identity')
        check(row['fixed'] is True and row['collision_enabled'] is True and row['native_collision_boxes'] == 1,
              'fixed native box collision retained')
        native = np.asarray(row['native_box_size'], dtype=float)
        check(native.shape == (7,) and np.isfinite(native).all() and np.array_equal(native[3:], np.zeros(4)),
              'exact native box encoding: three extents plus four zero padding values')
        check(np.allclose(native[:3], box['size_xyz'], atol=1e-7, rtol=0), 'actual collision dimensions')
        check(np.allclose(row['native_position'], box['centre_xyz'], atol=1e-7, rtol=0), 'actual static position')
        check(np.allclose(row['native_quaternion_wxyz'], [1., 0., 0., 0.], atol=1e-7, rtol=0), 'actual static rotation')
        check(np.allclose(row['surface_rgb'], palette[box['material_id']], atol=1e-7, rtol=0), 'installed surface color')


def audit_trial(directory, spec, supplied):
    check(supplied['scene_id'] == spec['scene_id'] and supplied['marker_case'] == spec['marker_case']
          and supplied['case_index'] == spec['case_index'], 'fixed trial identity')
    leaves = {*LEAVES, *(f'rgb_{i:04d}.png' for i in range(5))}
    check(set(supplied['artifact_sha256']) == leaves, 'exact artifact population')
    for name, sha in supplied['artifact_sha256'].items():
        check(digest(directory/name) == sha, f'artifact drift: {name}')
    raw = read_npz(directory/'physics_trace.npz')
    check(len(raw['timestamp_s']) == supplied['physics_samples'] == 950, 'complete physics population')
    check(np.allclose(raw['timestamp_s'], .002*np.arange(1, 951), atol=1e-10, rtol=0), 'physics clock')
    check(np.array_equal(raw['phase'], [0]*750+[1]*200), 'fixed zero-command phases')
    check(np.array_equal(raw['requested_command'], np.zeros((950, 3)))
          and np.array_equal(raw['applied_command'], np.zeros((950, 3))), 'only actual zero commands')
    topology = json.loads((directory/'contact_topology.json').read_text())
    flags, first = recompute_contact_flags(read_npz(directory/'native_contacts.npz'), topology, raw['timestamp_s'])
    check(np.array_equal(flags, raw['physics_contact'].astype(bool)), 'native contact accounting')
    check(first is None and not flags.any() and supplied['native_contact'] is False, 'contact-free stationary probe')
    check(supplied['stop_reason'] is None and supplied['completed_probe'] is True, 'no early physical stop')
    expected_objects = {'ground_plane', *(b['wall_id'] for b in spec['geometry']['wall_boxes'])}
    check(set(topology['environment_object_ids'].values()) == expected_objects, 'all physical objects in contact identity')
    check_static_objects(spec, json.loads((directory/'static_objects.json').read_text()))
    gains = json.loads((directory/'actuator_identity.json').read_text())
    check(json.loads((directory/'terminal_actuator_gains.json').read_text()) == gains['effective'], 'stable installed gains')
    check(np.allclose(gains['effective']['kp'], 20.) and np.allclose(gains['effective']['kv'], .5), 'checkpoint gains')
    sensors = reconstruct_sensors(raw)
    recorded = read_npz(directory/'ideal_sensor_samples.npz')
    check(set(recorded) == set(sensors) and len(recorded['measured_ns']) == supplied['sensor_samples'] == 95,
          'exact ordinary sensor population')
    for key in sensors:
        check(np.allclose(recorded[key], sensors[key], atol=1e-10, rtol=0), f'raw sensor replay: {key}')
    cameras = json.loads((directory/'camera_audit.json').read_text())
    decisions = json.loads((directory/'marker_decisions.json').read_text())
    check(len(cameras) == len(decisions) == supplied['rgb_packets'] == 5, 'five actual captures/decisions')
    observer = MarkerDiscovery()
    detected, discoveries = [], []
    for i, (camera, recorded_decision) in enumerate(zip(cameras, decisions, strict=True)):
        packet = load_continuation_observation(directory, i)
        now = 1_500_000_000+i*100_000_000
        sample = 749+i*50
        check(packet['image']['measured_ns'] == now and camera['physical_sample_index'] == sample
              and round(camera['timestamp_s']*1e9) == now and camera['rgb_file'] == f'rgb_{i:04d}.png', 'image clock/role')
        check(camera['rgb_sha256'] == hashlib.sha256(packet['image']['rgb'].tobytes()).hexdigest(), 'current pixel identity')
        check(camera['rigid_mount_no_obstacle_adjustment'] is True, 'no privileged camera adjustment')
        rotation = rotation_xyzw(raw['base_pose_world'][sample, 3:])
        expected = np.eye(4)
        expected[:3, :3] = np.stack([-rotation[:, 1], -rotation[:, 2], rotation[:, 0]], axis=1)
        expected[:3, 3] = raw['base_pose_world'][sample, :3]+rotation@ORIGIN_BODY
        check(np.allclose(camera['world_from_optical'], expected, atol=1e-7, rtol=0), 'actual rigid body camera')
        for schema in SCHEMAS:
            actual = packet['sensor_state'][schema.role][schema.name]
            for field, value in expected_history(raw, sensors, schema, now).items():
                check(np.allclose(actual[field], value, atol=1e-10, rtol=0), f'causal history: {i}/{schema.name}/{field}')
        result = observer.observe(packet, now_ns=now)
        check(recorded_decision == {'observation_index': i, 'pre_sample_index': sample, 'result': result},
              'exact detector and temporal discovery replay')
        detected.append(bool(result['detections']))
        if result['newly_discovered']:
            discoveries.append(i)
    check(detected == supplied['detected_frames'] and discoveries == supplied['discovery_indices']
          and result['distinct_marker_count'] == supplied['distinct_marker_count'], 'reported detector summary')
    expected_positive = spec['marker_case'] == 'positive'
    return {'scene_id': spec['scene_id'], 'audited_physics_samples': 950, 'audited_rgb_packets': 5,
            'detected_frames': detected, 'discovery_indices': discoveries,
            'matches_declared_response': detected == [expected_positive]*5
                and discoveries == ([2] if expected_positive else []),
            'expectation_is_evaluation_only': True}


def main():
    if len(sys.argv) != 1 or (OUTPUT/'raw_artifact_audit_v2.json').exists():
        raise ValueError('fixed one-shot audit; no overwrite')
    launch = json.loads((OUTPUT/'launch.json').read_text())
    report = json.loads((OUTPUT/'result.json').read_text())
    verify_bindings(launch['source_sha256'] | launch['input_sha256'] | launch['gait_sha256'])
    check(report['status'] == 'COMPLETE' and report['launch_sha256'] == digest(OUTPUT/'launch.json'), 'completed bound study')
    specs = trials()
    check(specs == launch['trial_specs'] and len(report['trials']) == 6, 'fixed six-case population')
    original = OUTPUT/'raw_artifact_audit.json'
    check(digest(original) == '1a99c1ce9ede3712849d26b554c1a5804954781dcab94cec53976ff020ccc2b9', 'preserved V1 failure identity')
    check(digest(OUTPUT/'launch.json') == '9514555d90fc360d07e0c7efe29afe8aa9047ddbab57295019e4a8dae321faa7'
          and digest(OUTPUT/'result.json') == '4300c237d9ff9ba3677ea58bc904faa56f42089e03a01e9ca3dd29e84520c019',
          'unchanged physical study identity')
    correction_sources = ('scripts/audit_go2_marker_beacon_development_v2.py',
                          'lewm/tests/test_marker_native_box_audit_correction_development.py',
                          'docs/go2_marker_native_box_audit_correction_development_2026-09-05.md')
    bindings = {p: digest(ROOT/p) for p in correction_sources}
    verify_bindings(bindings)
    write_json(OUTPUT/'audit_v2_binding.json', {'source_sha256': bindings,
        'study_launch_sha256': digest(OUTPUT/'launch.json'), 'study_result_sha256': digest(OUTPUT/'result.json'),
        'original_failed_audit_sha256': digest(original),
        'scope': 'strict seven-value native box decoding only; no physics, detector or outcome changes'})
    rows = []
    try:
        for spec, supplied in zip(specs, report['trials'], strict=True):
            row = audit_trial(OUTPUT/spec['scene_id'], spec, supplied); rows.append(row)
            print(json.dumps(row), flush=True)
        result = {'status': 'PASS', 'audited_trials': 6, 'audited_physics_samples': 5700,
                  'audited_rgb_packets': 30, 'trials': rows,
                  'all_declared_responses_match': all(r['matches_declared_response'] for r in rows),
                  'study_result_sha256': digest(OUTPUT/'result.json'),
                  'scope': 'physical marker acquisition evidence only; no navigation or hardware qualification'}
        verify_bindings(bindings)
        result['correction_binding_sha256'] = digest(OUTPUT/'audit_v2_binding.json')
        write_json(OUTPUT/'raw_artifact_audit_v2.json', result)
        print(json.dumps({'status': 'PASS', 'trials': 6, 'all_declared_responses_match': result['all_declared_responses_match']}), flush=True)
    except Exception as error:
        write_json(OUTPUT/'raw_artifact_audit_v2.json', {'status': 'FAIL', 'error': repr(error), 'trials': rows,
            'study_result_sha256': digest(OUTPUT/'result.json')})
        raise


if __name__ == '__main__':
    main()

