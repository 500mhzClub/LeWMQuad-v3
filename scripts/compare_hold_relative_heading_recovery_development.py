"""Measure navigation and executed recovery heading/clearance, using saved physics."""
import argparse
from collections import Counter
import json
import math

import numpy as np

from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.physical_execution_development import rotation_xyzw
from lewm.terminal_translation_pulse_development import command_sequences
from scripts.compare_continuous_navigation_arms_development import path, read, summarize


def wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


def recovery_evidence(root):
    plans = [p for p in read(root, 'planning.json') if 'selection' in p]
    active = [p for p in plans if p['selection'].get('hold_relative_heading_recovery', {}).get('applied')]
    frames = {f['frame']:f for f in read(root, 'native/in_memory_camera_observations.json')['frames']}
    requests = {r['simulator_ns']:r['requested_command'] for r in read(root, 'requests.json')}
    with np.load(root/'native/physics_trace.npz', allow_pickle=False) as arrays:
        physics = arrays['base_pose_world'].copy()
    boxes = []
    for box in read(root, 'native/camera_setup_identity.json')['environment']['physical_geometries']:
        if box['geom_type'] != 'BOX':
            continue
        quat = np.asarray(box['quaternion_world_wxyz'])
        rotation = rotation_xyzw(quat[[1,2,3,0]])
        if not np.allclose(rotation[2,:2], 0., atol=1e-6):
            raise ValueError('this centre-clearance evaluator requires vertical boxes')
        boxes.append((np.asarray(box['position_world_m']), rotation, np.asarray(box['data'][:2])/2))

    def clearance(points):
        return np.min([np.linalg.norm(np.maximum(np.abs(((points-centre)@rotation)[:,:2])-half, 0), axis=1)
            for centre, rotation, half in boxes], axis=0)

    rows = []
    for p in active:
        marker = p['selection']['hold_relative_heading_recovery']
        row = dict(frame=p['frame'], measured_ns=p['measured_ns'], on_time=p['on_time'],
            selected_recovery_action=marker['selected_action'], final_plan_action=p['action'],
            selection=marker, executed_recovery_sequence=False)
        rows.append(row)
        if p['action'] != marker['selected_action'] or p['frame']+7 not in frames:
            continue
        index = ACTIONS.index(p['action']); now = p['measured_ns']
        commands = command_sequences(p['committed_prefix'],
            pulse=bool(p['motion_correction'].get('terminal_translation_pulse', False)))[index,:7]
        times = [now+j*100_000_000+k*20_000_000 for j in range(7) for k in range(5)]
        if not all(t in requests for t in times) or not np.allclose(
                np.asarray([requests[t] for t in times]), np.repeat(commands,5,axis=0), rtol=0, atol=1e-8):
            continue
        f = p['frame']; indices = [frames[f+h]['physical_sample_index'] for h in (0,3,7)]
        origin, start, end = physics[indices]
        rotation = rotation_xyzw(origin[3:])
        points = ((physics[indices[1:],:3]-origin[:3])@rotation)[:,:2]
        yaws = [math.atan2(relative[1,0], relative[0,0]) for pose in (start,end)
            for relative in [rotation.T@rotation_xyzw(pose[3:])]]
        if 'scan_heading_error_rad' in p['selection']:
            target = p['selection']['scan_heading_error_rad']
            errors = [abs(wrap(target-yaw)) for yaw in yaws]
        else:
            waypoint = np.asarray(p['selection']['waypoint_body_xy_m'])
            errors = [abs(wrap(math.atan2(*(waypoint-point)[::-1])-yaw))
                for point,yaw in zip(points,yaws)]
        distances = clearance(physics[indices[1]:indices[2]+1,:3])
        forecast = np.asarray(p['motion_correction']['corrected_forecast_xy_m'])[index,6]
        row.update(executed_recovery_sequence=True,
            actual_heading_error_at_commit_start_rad=errors[0],
            actual_heading_error_at_commit_end_rad=errors[1],
            actual_heading_improvement_rad=errors[0]-errors[1],
            native_center_wall_clearance_at_commit_start_m=float(distances[0]),
            native_center_wall_clearance_at_commit_end_m=float(distances[-1]),
            native_center_wall_clearance_minimum_during_commit_m=float(distances.min()),
            corrected_endpoint_xy_error_m=float(np.linalg.norm(forecast-points[-1])))
    executed = [r for r in rows if r['executed_recovery_sequence']]
    return dict(plans=len(plans), actions=dict(Counter(p['action'] for p in plans)),
        recovery_activations=len(active), recovery_plans_on_time=sum(p['on_time'] for p in active),
        recovery_final_action_preserved=sum(p['action']==p['selection']['hold_relative_heading_recovery']['selected_action'] for p in active),
        executed_recovery_windows=len(executed),
        executed_heading_improving_windows=sum(r['actual_heading_improvement_rad']>0 for r in executed),
        native_center_wall_minimum_m=min((r['native_center_wall_clearance_minimum_during_commit_m'] for r in executed),default=None),
        rows=rows, native_state_evaluator_only=True,
        native_center_distance_is_not_articulated_clearance=True,
        heading_target_is_frozen_observed_target_at_plan_time=True,
        physical_heading_interval_ns=[300_000_000,700_000_000],
        overlapping_windows_not_independent=True, unexecuted_forecasts_scored=False,
        full_reserve_or_safety_certification_claimed=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=(0,1), required=True)
    i = parser.parse_args().layout_index
    baseline = path(f'go2_nearby_panorama_directed_native_layout{i:02d}_4800_v1_attempt_001')
    revised = path(f'go2_hold_relative_heading_recovery_native_layout{i:02d}_4800_v1_attempt_001')
    output = path(f'go2_hold_relative_heading_recovery_comparison_layout{i:02d}_v1_attempt_001')
    if output.exists():
        raise ValueError('preserve previous comparison')
    a,b = read(baseline,'launch.json'),read(revised,'launch.json')
    changed = sorted(k for k in a.keys()|b.keys() if a.get(k)!=b.get(k))
    if set(changed)-{'owner','experiment','comparison_condition','baseline_root_name','hold_relative_heading_recovery','extra_sources'}:
        raise ValueError(f'non-treatment settings differ: {changed}')
    if not all(b['extra_sources'].get(k)==v for k,v in a['extra_sources'].items()):
        raise ValueError('baseline source changed')
    evidence = recovery_evidence(revised)
    requests = [{p['simulator_ns']:p['requested_command'] for p in read(root,'requests.json')}
        for root in (baseline,revised)]
    first_difference = next((t for t in sorted(requests[0].keys()&requests[1].keys())
        if requests[0][t]!=requests[1][t]),None)
    first_recovery = min((r['measured_ns'] for r in evidence['rows']),default=None)
    report = dict(layout_index=i, comparison='hold_relative_preferred_heading_recovery',
        conditions=dict(baseline=summarize(baseline), recovery=summarize(revised)), mechanism=evidence,
        changed_launch_fields=changed, baseline_source_count=len(a['extra_sources']),
        added_source_paths=sorted(b['extra_sources'].keys()-a['extra_sources'].keys()),
        non_treatment_settings_and_baseline_sources_equal=True,
        first_requested_command_difference_ns=first_difference, first_recovery_measured_ns=first_recovery,
        execution_diverged_before_first_recovery=(first_difference<first_recovery
            if first_difference is not None and first_recovery is not None else None),
        causal_episode_improvement_established=False, repeatability_established=False)
    output.mkdir()
    with (output/'result.json').open('x') as f: json.dump(report,f,indent=2)
    print(json.dumps(dict(arrivals=report['conditions']['recovery']['independent_arrival_evaluation'],
        recovery_activations=evidence['recovery_activations'],executed_recovery_windows=evidence['executed_recovery_windows'],
        heading_improving_windows=evidence['executed_heading_improving_windows'])))


if __name__ == '__main__': main()
