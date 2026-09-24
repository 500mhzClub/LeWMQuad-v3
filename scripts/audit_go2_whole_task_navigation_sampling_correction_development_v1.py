#!/usr/bin/env python3
"""Full raw replay of continuous marker search and tentative remembered return."""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'lewm_genesis', ROOT/'lewm_worlds'):
    sys.path.insert(0, str(path))
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_sensor_state import SensorContractError
from lewm.fast_gyro_development import validate_fast_packet
from lewm.ground_projection_envelope_development import ORIGIN_BODY
from lewm.physical_execution_development import rotation_xyzw
from lewm.simulated_body_observation_development import SCHEMAS
from lewm.whole_task_navigation_development import WholeTaskNavigation
from lewm.whole_task_scene_development import trial_specs
from lewm.whole_task_metrics_development import reduce_whole_task, marker_centres_occluded
from lewm.whole_task_rgb_dataset_development import load_whole_task_observation as load_route_observation
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.run_go2_counterfactual_maze_dataset_development_v1 import array_binding
from scripts.run_go2_whole_task_navigation_sampling_correction_development_v1 import OUTPUT, LEAVES, digest, write_json, verify_bindings
from scripts.audit_go2_local_control_factorial_development_v1 import check, read_npz, recompute_contact_flags
from scripts.audit_go2_causal_rgb_body_capture_development_v1 import reconstruct_sensors, expected_history
from scripts.audit_go2_task_acquisition_continuation_development_v1 import scientific_decision


def check_static_objects(spec, rows):
    boxes = spec['geometry']['wall_boxes']
    check(len(boxes) == len(rows), 'exact native static population')
    palette = {'NEUTRAL_WALL': [.35, .35, .35], 'landmark_red': [.85, .12, .08], 'landmark_blue': [.10, .22, .85]}
    for box, row in zip(boxes, rows, strict=True):
        expected = {'object_id': box['wall_id'], 'kind': 'wall', 'center_xyz_m': box['centre_xyz'],
                    'size_xyz_m': box['size_xyz'], 'yaw_rad': box['yaw_rad'], 'material_id': box['material_id'],
                    'roll_rad': 0., 'pitch_rad': 0.}
        check(row['pack_object'] == expected and row['native_name'] == box['wall_id'], 'static construction identity')
        check(row['fixed'] is True and row['collision_enabled'] is True and row['native_collision_boxes'] == 1,
              'native fixed box collision')
        native = np.asarray(row['native_box_size'], dtype=float)
        check(native.shape == (7,) and np.isfinite(native).all() and np.array_equal(native[3:], np.zeros(4)), 'native box padding')
        check(np.allclose(native[:3], box['size_xyz'], atol=1e-7, rtol=0), 'actual box extents')
        # Genesis uses float32 native positions; retain the exact decimal pack
        # binding above and compare readback against its native representation.
        position = np.asarray(box['centre_xyz'], dtype=np.float32).astype(float)
        check(np.allclose(row['native_position'], position, atol=1e-7, rtol=0), 'actual box position')
        q = np.array([math.cos(box['yaw_rad']/2), 0., 0., math.sin(box['yaw_rad']/2)])
        actual = np.asarray(row['native_quaternion_wxyz'])
        check(np.allclose(actual, q, atol=1e-7, rtol=0) or np.allclose(actual, -q, atol=1e-7, rtol=0), 'actual box rotation')
        check(np.allclose(row['surface_rgb'], palette[box['material_id']], atol=1e-7, rtol=0), 'actual marker/wall color')


def audit_trial(directory, spec, row, template=None):
    check(all(row[k] == spec[k] for k in ('scene_id', 'case_index', 'method', 'memory_arm', 'layout_name')), 'scan identity')
    check(1 <= row['rgb_packets'] <= 3606, 'scan RGB budget')
    required = set(LEAVES) | {f'rgb_{i:04d}.png' for i in range(row['rgb_packets'])}
    check(set(row['artifact_sha256']) == required, 'exact artifact population')
    for name, expected in row['artifact_sha256'].items():
        check(digest(directory / name) == expected, f'artifact binding {name}')
    gains = json.loads((directory / 'actuator_identity.json').read_text())
    check(gains['effective'] == json.loads((directory / 'terminal_actuator_gains.json').read_text())
          == {'kp': [20.] * 12, 'kv': [.5] * 12}, 'effective gains')
    raw = read_npz(directory / 'physics_trace.npz')
    count, start = len(raw['timestamp_s']), row['prefix_terminal_sample_index']
    check(count == row['physics_samples'] and 0 < count <= 181000 and start == min(count, 750) - 1, 'physical population')
    check(all(np.isfinite(v).all() for v in raw.values()), 'finite physical trace')
    check(np.allclose(raw['timestamp_s'], .002 * np.arange(1, count + 1), rtol=0, atol=1e-10), 'physics clock')
    check(np.all(raw['phase'][:start + 1] == 0) and np.all(raw['requested_command'][:start + 1] == 0)
          and np.all(raw['applied_command'][:start + 1] == 0), 'settling commands')
    check(array_binding({k: v[:start + 1] for k, v in raw.items()}) == row['prefix_binding']['physics_arrays'], 'settling raw binding')
    histories = read_npz(directory / 'policy_histories.npz')
    packet_index = row['branch_start_observation_index']
    check(array_binding({k: v[packet_index] for k, v in histories.items()}) == row['prefix_binding']['history_arrays'], 'settling histories')
    topology = json.loads((directory / 'contact_topology.json').read_text())
    check(set(topology['environment_object_ids'].values()) == {'ground_plane'} | {w['wall_id'] for w in spec['geometry']['wall_boxes']}, 'scene identity')
    flags, first = recompute_contact_flags(read_npz(directory / 'native_contacts.npz'), topology, raw['timestamp_s'])
    check(np.array_equal(flags, raw['physics_contact'].astype(bool)), 'native contact flags')
    stop_reason = row['response']['stop_reason']
    if first is not None:
        check(first['sample_index'] == count - 1 and stop_reason == 'DISALLOWED_CONTACT', 'first-contact immediate stop')
    else:
        check(stop_reason in (None, 'BODY_STABILITY_LIMIT'), 'unsupported physical stop')
    decisions = json.loads((directory / 'task_decisions.json').read_text())
    tape = json.loads((directory / 'command_tape.json').read_text())
    geometry = ArticulatedCollisionGeometry(URDF)
    controller = WholeTaskNavigation(spec['method'], geometry, memory_arm=spec['memory_arm'])
    active, terminal = [], None
    for tick, decision in enumerate(decisions):
        check(tick <= 3600 and decision['tick'] == tick and decision['pre_sample_index'] == start + 50 * tick, 'decision sequence')
        packet = load_route_observation(directory, decision['observation_index'])
        check(packet['sensor_state']['decision_ns'] == decision['decision_ns']
              == int(round(raw['timestamp_s'][decision['pre_sample_index']] * 1e9)), 'decision clock')
        fast = load_fast_packet(directory, decision['observation_index'])
        validate_fast_packet(fast, packet, now_ns=decision['decision_ns'])
        expected = controller.observe(packet, fast, now_ns=decision['decision_ns'])
        check(scientific_decision(expected) == scientific_decision(decision['controller']),
              'exact actual-packet controller and model replay')
        done = expected['terminal']
        check(decision['executed'] == (not done), 'execution marker')
        if done:
            check(tick == len(decisions) - 1, 'decision after controller terminal')
            terminal = expected['status']
        else:
            active.append({'phase': 1, 'pre_sample_index': decision['pre_sample_index'], 'requested_command': expected['requested_command']})
    fault = row['response']['sensor_fault']
    if fault is not None:
        check(terminal is None and fault['tick'] == len(decisions) and fault['pre_sample_index'] == start + 50 * len(decisions), 'fault clock')
        packet = load_route_observation(directory, fault['observation_index'])
        try:
            controller.observe(packet, load_fast_packet(directory, fault['observation_index']), now_ns=packet['sensor_state']['decision_ns'])
        except SensorContractError as error:
            check(str(error) == fault['reason'], 'fault reason replay')
        else:
            raise ValueError('recorded sensor fault did not replay')
        terminal = 'FAILED_SENSOR'
    check(tape[:len(active)] == active, 'decision command tape')
    release = tape[len(active):]
    check(len(release) <= 5 and all(e['phase'] == 2 and e['requested_command'] == [0., 0., 0.] for e in release), 'release tape')
    check(not release or terminal is not None, 'release before terminal')
    reported_terminal = terminal or ('PHYSICAL_STOP' if stop_reason is not None else controller.status)
    check(reported_terminal == row['response']['controller_terminal'], 'terminal summary')
    previous = start
    for entry in tape:
        check(entry['pre_sample_index'] == previous, 'command order')
        size = min(50, count - previous - 1)
        check(size > 0, 'empty executed command')
        sl = slice(previous + 1, previous + 1 + size)
        request = entry['requested_command']
        check(np.all(raw['phase'][sl] == entry['phase']) and np.array_equal(raw['requested_command'][sl], np.tile(request, (size, 1))), 'requested command/phase')
        applied = raw['applied_command'][previous] + np.clip(np.array(request, dtype=np.float32) - raw['applied_command'][previous], [-.25, 0, -.35], [.25, 0, .35])
        check(np.allclose(raw['applied_command'][sl], applied, rtol=0, atol=1e-7), 'applied slew')
        previous += size
    check(previous == count - 1, 'unexplained physics after tape')
    if stop_reason is None:
        check(terminal is not None and len(release) == 5, 'incomplete nonstopped scan')
    if stop_reason is not None:
        controller.finish_physical_stop(now_ns=int(round(raw['timestamp_s'][-1]*1e9)))
    check(controller.ledgers() == json.loads((directory/'task_ledgers.json').read_text()), 'exact provisional ledgers')
    check(json.loads(json.dumps(controller.memory_snapshot(), allow_nan=False)) == json.loads((directory/'task_memory.json').read_text()), 'exact terminal episodic memory')
    check_static_objects(spec, json.loads((directory/'static_objects.json').read_text()))
    initial_camera = json.loads((directory/'camera_audit.json').read_text())[0]
    occluded = marker_centres_occluded(spec, np.asarray(initial_camera['world_from_optical'])[:3, 3])
    check(reduce_whole_task(raw, start, decisions, terminal=reported_terminal, stop_reason=stop_reason,
                           sensor_fault=fault, initial_marker_occluded=occluded) == row['response'],
          'independent physical whole-task outcome reduction')
    sensors = reconstruct_sensors(raw)
    recorded = read_npz(directory / 'ideal_sensor_samples.npz')
    check(set(recorded) == set(sensors) and len(sensors['measured_ns']) == row['sensor_samples'], 'sensor population')
    for key, value in sensors.items():
        check(recorded[key].shape == value.shape and np.allclose(recorded[key], value, rtol=0, atol=1e-10), 'causal sensor reconstruction')
    fast_samples = read_npz(directory / 'fast_gyro_samples.npz')
    fast_histories = read_npz(directory / 'fast_gyro_histories.npz')
    expected_ns = np.rint(raw['timestamp_s'] * 1e9).astype(np.int64)
    expected_rates = np.stack([rotation_xyzw(p[3:]).T @ w[3:] for p, w in zip(raw['base_pose_world'], raw['base_twist_world'], strict=True)])
    check(set(fast_samples) == {'measured_ns', 'available_ns', 'values', 'valid'} == set(fast_histories), 'fast array names')
    check(row['fast_gyro_samples'] == count and fast_samples['values'].shape == (count, 3), 'all native-rate measurements')
    check(np.array_equal(fast_samples['measured_ns'], expected_ns) and np.array_equal(fast_samples['available_ns'], expected_ns), 'fast acquisition clock')
    check(fast_samples['valid'].dtype == bool and fast_samples['valid'].shape == (count, 3) and fast_samples['valid'].all(), 'fast validity')
    check(np.allclose(fast_samples['values'], expected_rates, rtol=0, atol=1e-12), 'independent virtual gyro reconstruction')
    check(all(len(v) == row['rgb_packets'] for v in fast_histories.values()), 'fast history population')
    cameras = json.loads((directory / 'camera_audit.json').read_text())
    check(len(cameras) == row['rgb_packets'], 'camera population')
    expected_times = {float(raw['timestamp_s'][start]), float(raw['timestamp_s'][-1])}
    expected_times |= {float(raw['timestamp_s'][e['pre_sample_index']]) for e in [*tape, *decisions]}
    if fault is not None:
        expected_times.add(float(raw['timestamp_s'][fault['pre_sample_index']]))
    check([c['timestamp_s'] for c in cameras] == sorted(expected_times), 'command/terminal camera coverage')
    for index, camera in enumerate(cameras):
        packet = load_route_observation(directory, index)
        ns = packet['image']['measured_ns']
        check(ns == int(round(camera['timestamp_s'] * 1e9)) and raw['timestamp_s'][camera['physical_sample_index']] == camera['timestamp_s'], 'image clock')
        check(hashlib.sha256(packet['image']['rgb'].tobytes()).hexdigest() == camera['rgb_sha256'], 'RGB binding')
        selected = np.flatnonzero((expected_ns <= ns) & (expected_ns >= ns - 100_000_000))[-51:]
        expected_fast = {'values': np.zeros((51, 3)), 'valid': np.zeros((51, 3), bool),
                         'measured_ns': np.full(51, -1, np.int64), 'available_ns': np.full(51, -1, np.int64)}
        expected_fast['values'][-len(selected):] = expected_rates[selected]
        expected_fast['valid'][-len(selected):] = True
        expected_fast['measured_ns'][-len(selected):] = expected_ns[selected]
        expected_fast['available_ns'][-len(selected):] = expected_ns[selected]
        for field, expected in expected_fast.items():
            check(np.array_equal(fast_histories[field][index], expected), 'exact causal fast history')
        pose = raw['base_pose_world'][camera['physical_sample_index']]
        rotation = rotation_xyzw(pose[3:])
        transform = np.asarray(camera['world_from_optical'])
        optical_to_body = np.array([[0., 0, 1], [-1., 0, 0], [0., -1., 0]])
        check(np.allclose(rotation @ optical_to_body, transform[:3, :3], rtol=0, atol=1e-8)
              and np.allclose(pose[:3] + rotation @ ORIGIN_BODY, transform[:3, 3], rtol=0, atol=1e-8), 'actual mounted camera transform')
        for schema in SCHEMAS:
            expected = expected_history(raw, sensors, schema, ns)
            actual = packet['sensor_state'][schema.role][schema.name]
            check(all(np.allclose(actual[k], value, rtol=0, atol=1e-10) for k, value in expected.items()), 'causal history')
    return {'scene_id': spec['scene_id'], 'status': 'PASS', 'rgb_packets': len(cameras),
            'decisions_replayed': len(decisions), 'method': spec['method'], 'fast_samples': count,
            'physical_task_success': row['response']['physical_task_success'],
            'memory_arm': spec['memory_arm'], 'sensor_fault': fault}


def main():
    if len(sys.argv) != 1 or (OUTPUT/'raw_artifact_audit.json').exists():
        raise ValueError('fixed fresh whole-task audit required')
    launch = json.loads((OUTPUT/'launch.json').read_text())
    report = json.loads((OUTPUT/'result.json').read_text())
    check(report['status'] == 'COMPLETE' and report['completed_trials'] == report['planned_trials'] == 4
          and len(report['trials']) == 4, 'complete fixed whole-task population')
    check(report['launch_sha256'] == digest(OUTPUT/'launch.json') and launch['trial_specs'] == trial_specs(), 'launch identity')
    verify_bindings(launch['source_sha256'] | launch['input_sha256'] | launch['gait_sha256'])
    rows, prefixes = [], {}
    try:
        for spec, row in zip(trial_specs(), report['trials'], strict=True):
            directory = OUTPUT/spec['scene_id']
            check(json.loads((directory/'result.json').read_text()) == row, 'root/member identity')
            rows.append(audit_trial(directory, spec, row))
            reference = prefixes.setdefault(spec['case_index'], row['prefix_binding'])
            for key in ('physics_arrays', 'history_arrays', 'physics_samples', 'timestamp_ns'):
                check(reference[key] == row['prefix_binding'][key], 'paired actual settled prefix')
            print(json.dumps({'event': 'task_audited', 'completed': len(rows), 'total': 4,
                              'decisions': rows[-1]['decisions_replayed'], 'physical_task_success': rows[-1]['physical_task_success']}), flush=True)
        verify_bindings(launch['source_sha256'] | launch['input_sha256'] | launch['gait_sha256'])
        write_json(OUTPUT/'raw_artifact_audit.json', {'status': 'PASS', 'audited_trials': 4, 'trials': rows,
            'study_result_sha256': digest(OUTPUT/'result.json'), 'audit_source_sha256': digest(Path(__file__)),
            'scope': 'full continuous development sensor/controller/memory and physical outcome replay; no final qualification'})
        print(json.dumps({'status': 'PASS', 'audited_trials': 4, 'decisions_replayed': sum(r['decisions_replayed'] for r in rows)}), flush=True)
    except Exception as error:
        write_json(OUTPUT/'raw_artifact_audit.json', {'status': 'FAIL', 'error': repr(error), 'trials': rows,
            'study_result_sha256': digest(OUTPUT/'result.json')})
        raise


if __name__ == '__main__':
    main()
