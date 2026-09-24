#!/usr/bin/env python3
"""Fixed gravity-feedback comparison on all eight existing development routes."""
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'lewm_genesis', ROOT / 'lewm_worlds'):
    sys.path.insert(0, str(path))
from lewm.gravity_feedback_ground_development import CausalGravityFeedbackGround, MODES
from lewm.causal_sensor_state import SensorContractError
from lewm.ground_projection_envelope_development import observe_ground_envelope, project_ground_rays
from lewm.floor_visibility_reference_development import visible_floor
from lewm.physical_execution_development import rotation_xyzw
from lewm.route_rgb_dataset_development import load_route_observation
from scripts.analyze_go2_ground_projection_envelope_development_v1 import (
    OUTPUT as PREVIOUS, ROUTES, route_spec, MOTIFS, WIDTHS, verify_bindings, digest, write_json)

OUTPUT = ROOT / '.generated/go2_gravity_feedback_ground_development_v1_attempt_001'
NEW_SOURCES = ('lewm/gravity_feedback_ground_development.py',
               'lewm/tests/test_gravity_feedback_ground_development.py',
               str(Path(__file__).relative_to(ROOT)),
               'lewm/tests/test_gravity_feedback_ground_raw_audit.py',
               'docs/go2_gravity_feedback_ground_development_v1_2026-09-05.md')


def frame_metrics(state, rays, truth, pose):
    visible = truth['valid'] & truth['visible_floor']
    count = int(visible.sum())
    if state is None:
        return {'sensor_available': False, 'visible_floor': count, 'nominal_points': 0,
                'unavailable_floor_points': count, 'error_sum_m': 0., 'error_max_m': None,
                'quarter_m_failure_points': count, 'normal_error_rad': None,
                'signed_height_error_m': None, 'feedback_applied': False, 'feedback_reason': 'latched_sensor_failure'}
    normal = np.asarray(state['up_current_body'])
    up = rotation_xyzw(pose[3:])[2]
    angle = math.atan2(float(np.linalg.norm(np.cross(normal, up))), float(normal @ up))
    prediction = project_ground_rays(normal, state['body_origin_height_m'], rays, height_radius=0., angle_radius=0.)
    selected = visible & prediction['nominal_valid']
    error = np.abs(prediction['nominal_optical_depth_m'][selected] - truth['ground_optical_depth_m'][selected]) * np.linalg.norm(rays[selected], axis=-1)
    missing = int((visible & ~prediction['nominal_valid']).sum())
    return {'sensor_available': True, 'visible_floor': count, 'nominal_points': int(error.size),
            'unavailable_floor_points': missing, 'error_sum_m': float(error.sum()),
            'error_max_m': float(error.max()) if error.size else None,
            'quarter_m_failure_points': missing + int((error > .25).sum()),
            'normal_error_rad': angle, 'signed_height_error_m': state['body_origin_height_m'] - float(pose[2]),
            'feedback_applied': state['feedback_applied'], 'feedback_reason': state['feedback']['reason']}


def summarize(rows):
    if not rows:
        raise ValueError('complete source frames, including unavailable outputs, required')
    totals = {k: sum(r[k] for r in rows) for k in ('visible_floor', 'nominal_points', 'unavailable_floor_points',
                                                 'error_sum_m', 'quarter_m_failure_points')}
    valid = [r for r in rows if r['sensor_available']]
    errors = [r['error_max_m'] for r in rows if r['error_max_m'] is not None]
    return {**totals, 'frames': len(rows), 'sensor_available_frames': len(valid),
            'feedback_applied_frames': sum(r['feedback_applied'] for r in rows),
            'feedback_reasons': {reason: sum(r['feedback_reason'] == reason for r in rows)
                                 for reason in sorted({r['feedback_reason'] for r in rows})},
            'normal_error_mean_rad': float(np.mean([r['normal_error_rad'] for r in valid])) if valid else None,
            'normal_error_max_rad': max(r['normal_error_rad'] for r in valid) if valid else None,
            'height_error_mae_m': float(np.mean([abs(r['signed_height_error_m']) for r in valid])) if valid else None,
            'point_error_mean_m': totals['error_sum_m'] / totals['nominal_points'] if totals['nominal_points'] else None,
            'point_error_max_m': max(errors) if errors else None,
            'quarter_m_failure_fraction': totals['quarter_m_failure_points'] / totals['visible_floor'] if totals['visible_floor'] else None,
            'normal_metrics_complete': len(valid) == len(rows)}


def main():
    if len(sys.argv) != 1 or OUTPUT.exists() or OUTPUT.resolve() != OUTPUT:
        raise ValueError('fixed fresh gravity-feedback comparison required')
    inputs = {str((PREVIOUS / 'launch.json').relative_to(ROOT)): '47923287b16fc6e9bfb2c33d05f821d1c1e90ff760c72224305cab19671f42f3',
              str((PREVIOUS / 'result.json').relative_to(ROOT)): 'c31d1c3e9d747d9a03e396286f47aad47e2dd6bd9f10106e3e7d6cf7ef7464a9'}
    verify_bindings(inputs)
    previous_launch = json.loads((PREVIOUS / 'launch.json').read_text())
    inputs.update(previous_launch['input_sha256'])
    sources = previous_launch['source_sha256'] | {p: digest(ROOT / p) for p in NEW_SOURCES}
    verify_bindings(sources | inputs)
    previous_result = json.loads((PREVIOUS / 'result.json').read_text())
    if previous_result['status'] != 'COMPLETE' or previous_result['frames_processed'] != 1498:
        raise ValueError('completed baseline ray population required')
    previous_rows = {r['scene_id']: r for r in previous_result['trials']}
    report = json.loads((ROUTES / 'result.json').read_text())
    specs = {s['scene_id']: s for s in (route_spec(m, w) for m in MOTIFS for w in WIDTHS)}
    if (report['status'] != 'COMPLETE' or report['completed_trials'] != 8
            or {m['scene_id'] for m in report['trials']} != set(specs)
            or json.loads((ROUTES / 'launch.json').read_text())['trial_specs'] != list(specs.values())):
        raise ValueError('fixed actual route specs required')
    OUTPUT.mkdir()
    write_json(OUTPUT / 'launch.json', {'source_sha256': sources, 'input_sha256': inputs,
               'modes': MODES, 'frames_expected': 1498, 'feedback_time_constant_s': 2.,
               'gates': {'mean_gravity_m_s2': 9.81, 'mean_magnitude_tolerance_m_s2': .75,
                         'force_residual_rms_max_m_s2': 3., 'last_five_command_component_range_max': .05},
               'scope': 'fixed no-fit causal attitude/point observation comparison on reused routes, not yaw or safety calibration'})
    trials = []
    try:
        for member in report['trials']:
            scene = member['scene_id']
            directory = ROUTES / scene
            names = ['physics_trace.npz', 'camera_audit.json', 'policy_histories.npz', 'policy_observations.json']
            names.extend(f'rgb_{i:04d}.png' for i in range(member['rgb_packets']))
            verify_bindings({str((directory / name).relative_to(ROOT)): member['artifact_sha256'][name] for name in names})
            with np.load(directory / 'physics_trace.npz', allow_pickle=False) as archive:
                poses = archive['base_pose_world']
            cameras = json.loads((directory / 'camera_audit.json').read_text())
            models = {mode: CausalGravityFeedbackGround(mode) for mode in MODES}
            faults, rows = {}, []
            for index, camera in enumerate(cameras):
                packet = load_route_observation(directory, index)
                now = packet['image']['measured_ns']
                states = {}
                for mode, model in models.items():
                    if mode in faults:
                        states[mode] = None
                        continue
                    try:
                        states[mode] = model.begin(packet, now_ns=now) if index == 0 else model.step(packet, now_ns=now)
                    except SensorContractError as error:
                        faults[mode] = {'frame': index, 'decision_ns': now, 'reason': str(error)}
                        states[mode] = None
                if states['gyro_only'] is None:
                    raise ValueError('unchanged previously complete baseline unexpectedly unavailable')
                envelope = observe_ground_envelope(packet, states['gyro_only'], now_ns=now, height_radius=.03, angle_radius=.1)
                truth = visible_floor(camera['world_from_optical'], specs[scene]['geometry']['wall_boxes'])
                pose = poses[camera['physical_sample_index']]
                metrics = {mode: frame_metrics(state, envelope['rays_body'], truth, pose) for mode, state in states.items()}
                witness = previous_rows[scene]['frames'][index]
                prior = witness['families']['0.01:0.025']
                baseline = metrics['gyro_only']
                if (witness['decision_ns'] != now or baseline['visible_floor'] != prior['visible_floor']
                        or baseline['nominal_points'] != prior['nominal_visible_points']
                        or baseline['error_sum_m'] != prior['nominal_error_sum_m']
                        or baseline['error_max_m'] != prior['nominal_error_max_m']):
                    raise ValueError('exact frozen baseline ray replay changed')
                rows.append({'observation_index': index, 'decision_ns': now, 'metrics': metrics})
            trials.append({'scene_id': scene, 'source_task_success': member['status'] == 'SUCCESS',
                           'frames': rows, 'faults': faults,
                           'modes': {mode: summarize([r['metrics'][mode] for r in rows]) for mode in MODES}})
            print(json.dumps({'event': 'gravity_route_compared', 'completed': len(trials), 'planned': 8,
                              'frames': len(rows), 'faults': faults}), flush=True)
        if sum(len(t['frames']) for t in trials) != 1498:
            raise ValueError('full frame population changed')
        paired = {}
        for mode in MODES[1:]:
            paired[mode + '_minus_gyro_only'] = {}
            for key in ('normal_error_mean_rad', 'point_error_mean_m', 'quarter_m_failure_fraction'):
                complete = all(t['modes'][m]['normal_metrics_complete'] for t in trials for m in ('gyro_only', mode))
                values = [t['modes'][mode][key] - t['modes']['gyro_only'][key] for t in trials] if complete else None
                paired[mode + '_minus_gyro_only'][key] = {'all_routes_complete': complete,
                    'per_route_delta': values, 'route_macro_delta': float(np.mean(values)) if values is not None else None}
        verify_bindings(sources | inputs)
        write_json(OUTPUT / 'result.json', {'status': 'COMPLETE', 'trials': trials,
                   'paired_descriptive_comparisons': paired, 'frames': 1498,
                   'launch_sha256': digest(OUTPUT / 'launch.json'), 'ground_plane_qualified': False,
                   'scope': 'reused correlated routes and ideal sensors; no fitted gate, yaw/clearance qualification or navigation execution'})
        print(json.dumps({'status': 'COMPLETE', 'routes': 8, 'frames': 1498}), flush=True)
    except Exception as error:
        write_json(OUTPUT / 'result.json', {'status': 'FAIL', 'error': repr(error), 'trials': trials,
                   'launch_sha256': digest(OUTPUT / 'launch.json')})
        raise


if __name__ == '__main__':
    main()
