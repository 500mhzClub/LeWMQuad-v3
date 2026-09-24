"""Evaluation-only per-leg crossing/release windows and continuous task result."""
import copy

import numpy as np

from lewm.observed_traversal_metrics_development import reduce_traversal


def reduce_continuation(spec, raw, start, decisions, terminal, stop_reason, sensor_fault, model):
    legs = []
    for leg_index, stage in enumerate(('FIRST', 'SECOND')):
        rows = [r for r in decisions if r['controller']['stage'] == stage and r['controller']['child'] is not None]
        if not rows:
            legs.append({'leg_index': leg_index, 'stage_started': False, 'response': None})
            continue
        begin = rows[0]['pre_sample_index']
        tail = rows[-1]
        child = tail['controller']['child']
        child_terminal = child['status'] if child['terminal'] else None
        # First leg's release is the first500 ms of the actual zero hold,
        # not the later scan or second destination. No extra physics is run.
        end = min(len(raw['timestamp_s'])-1, tail['pre_sample_index']+250) if child_terminal else len(raw['timestamp_s'])-1
        view = {name: values[begin:end+1].copy() for name, values in raw.items()}
        view['phase'][:] = 1
        if child_terminal: view['phase'][tail['pre_sample_index']-begin+1:] = 2
        selected = [{'pre_sample_index': r['pre_sample_index']-begin,
                     'controller': r['controller']['child']} for r in rows]
        evaluation_spec = {'geometry': spec['evaluation_leg_geometries'][leg_index]}
        leg_stop = stop_reason if end == len(raw['timestamp_s'])-1 else None
        # A later fault does not retroactively invalidate a completed first
        # crossing. The overall task still fails on every fault or native stop.
        leg_fault = sensor_fault if sensor_fault is not None and begin <= sensor_fault['pre_sample_index'] <= end else None
        response = reduce_traversal(evaluation_spec, view, 0, selected, child_terminal, leg_stop, leg_fault, model)
        release_indices = np.flatnonzero(view['phase'] == 2)
        zero_requests = bool(len(release_indices) == 250 and np.all(view['requested_command'][release_indices] == 0.))
        response['actual_zero_release_requests'] = zero_requests
        response['integration_success'] = response['integration_success'] and zero_requests
        legs.append({'leg_index': leg_index, 'stage_started': True, 'start_sample_index': begin,
                     'end_sample_index': end, 'response': response})
    chosen = [r['controller']['selected_side_branch'] for r in decisions if r['controller']['selected_side_branch'] is not None]
    if len(chosen) > 1: raise ValueError('one selected side branch maximum')
    scan_rows = [r for r in decisions if r['controller']['scan'] is not None]
    scan_drift = None
    if scan_rows:
        begin, end = scan_rows[0]['pre_sample_index'], scan_rows[-1]['pre_sample_index']
        scan_drift = float(np.linalg.norm(raw['base_pose_world'][begin:end+1, :2]-raw['base_pose_world'][begin, :2], axis=1).max())
    checks = {'controller_completed': terminal == 'COMPLETE_PROVISIONAL',
              'first_leg_integration': bool(legs[0]['response'] and legs[0]['response']['integration_success']),
              'second_leg_integration': bool(legs[1]['response'] and legs[1]['response']['integration_success']),
              'completed_scan': bool(scan_rows and scan_rows[-1]['controller']['scan']['status'] == 'COMPLETE'),
              'observed_side_branch_selected': bool(chosen),
              'no_native_stop': stop_reason is None, 'no_contact': not bool(raw['physics_contact'].any()),
              'no_sensor_fault': sensor_fault is None}
    return {'controller_terminal': terminal, 'stop_reason': stop_reason, 'sensor_fault': sensor_fault,
            'checks': checks, 'two_leg_integration_success': all(checks.values()), 'legs': legs,
            'selected_side_branch': copy.deepcopy(chosen[0]) if chosen else None,
            'maximum_scan_xy_drift_m': scan_drift,
            'total_post_settle_seconds': float(raw['timestamp_s'][-1]-raw['timestamp_s'][start]),
            'trusted_graph_edges': 0,
            'scope': 'two continuous development traversals; no place recognition, beacon return, independent-maze or hardware qualification'}
