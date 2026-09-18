"""Compare saved waypoint-heading forecasts with matching executed native motion."""
import argparse
from collections import Counter
import json
import numpy as np

from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.physical_execution_development import rotation_xyzw
from lewm.terminal_translation_pulse_development import command_sequences
from scripts.navigation_artifact_root_development import BASE, validate_root


def scalar_stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values): return dict(count=0)
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
        p90=float(np.percentile(values, 90)), minimum=float(values.min()), maximum=float(values.max()))


def evaluate(root, *, start_frame, end_frame):
    validate_root(root)
    read = lambda name: json.loads((root/name).read_text())
    frames = {r['frame']:r for r in read('native/in_memory_camera_observations.json')['frames']}
    requests = {r['simulator_ns']:r['requested_command'] for r in read('requests.json')}
    with np.load(root/'native/physics_trace.npz', allow_pickle=False) as arrays:
        physics = arrays['base_pose_world'].copy()
    plans = [r for r in read('planning.json') if start_frame <= r['frame'] <= end_frame
        and 'motion_correction' in r and 'selection' in r]
    rows = []; transitions = Counter(); target_steps = []; previous_target = None
    for p in plans:
        s = p['selection']; frame = p['frame']; now = p['measured_ns']
        transitions[f"{s.get('before_memory_filter_action')} -> {p['action']}"] += 1
        target = p.get('lookahead', {}).get('target_map_xy_m')
        if target is not None:
            if previous_target is not None:
                target_steps.append(float(np.linalg.norm(np.asarray(target)-previous_target)))
            previous_target = np.asarray(target)
        else:
            previous_target = None
        if p['action'] not in ('left_turn', 'right_turn') or 'scan_heading_error_rad' in s:
            continue
        index = ACTIONS.index(p['action']); candidate = s['candidates'][index]
        if 'predicted_heading_error_at_commit_end_rad' not in candidate or frame+7 not in frames:
            continue
        pulse = bool(p['motion_correction'].get('terminal_translation_pulse', False))
        commands = command_sequences(p['committed_prefix'], pulse=pulse)[index]
        if not all(t in requests and np.allclose(requests[t], command, rtol=0, atol=1e-8)
                for j, command in enumerate(commands[:7])
                for t in range(now+j*100_000_000, now+(j+1)*100_000_000, 20_000_000)):
            continue
        origin = physics[frames[frame]['physical_sample_index']]
        R = rotation_xyzw(origin[3:]); goal = np.asarray(s['waypoint_body_xy_m'])
        errors = []; yaw = []; displacements = []
        for h in (3, 7):
            pose = physics[frames[frame+h]['physical_sample_index']]
            delta = (R.T@(pose[:3]-origin[:3]))[:2]
            relative = R.T@rotation_xyzw(pose[3:])
            angle = float(np.arctan2(relative[1, 0], relative[0, 0]))
            bearing = float(np.arctan2(*(goal-delta)[::-1]))
            errors.append(float(abs(np.arctan2(np.sin(bearing-angle), np.cos(bearing-angle)))))
            yaw.append(angle); displacements.append(delta)
        predicted_gain = candidate['predicted_heading_error_at_commit_start_rad']-candidate['predicted_heading_error_at_commit_end_rad']
        actual_gain = errors[0]-errors[1]
        rows.append(dict(frame=frame, action=p['action'],
            predicted_heading_gain_rad=predicted_gain, actual_heading_gain_rad=actual_gain,
            actual_commit_yaw_rad=float(np.arctan2(np.sin(yaw[1]-yaw[0]), np.cos(yaw[1]-yaw[0]))),
            actual_commit_translation_m=float(np.linalg.norm(displacements[1]-displacements[0])),
            predicted_heading_error_end_rad=candidate['predicted_heading_error_at_commit_end_rad'],
            actual_heading_error_end_rad=errors[1],
            memory_changed_action=s.get('before_memory_filter_action') != p['action']))
    return dict(start_frame=start_frame, end_frame=end_frame, plans=len(plans),
        selected_action_transitions=dict(transitions), route_target_step_m=scalar_stats(target_steps),
        exact_route_target_repeats=sum(x == 0 for x in target_steps),
        matched_executed_waypoint_turns=len(rows),
        predicted_improvement_count=sum(r['predicted_heading_gain_rad'] > 0 for r in rows),
        actual_improvement_count=sum(r['actual_heading_gain_rad'] > 0 for r in rows),
        predicted_improvement_but_actual_worsening_count=sum(r['predicted_heading_gain_rad'] > 0
            and r['actual_heading_gain_rad'] < 0 for r in rows),
        heading_gain_error_rad=scalar_stats([r['predicted_heading_gain_rad']-r['actual_heading_gain_rad'] for r in rows]),
        actual_commit_yaw_magnitude_rad=scalar_stats([abs(r['actual_commit_yaw_rad']) for r in rows]),
        native_state_evaluator_only=True, observed_waypoint_anchored_in_actual_start_body=True,
        actual_requested_prefix_matched_through_700ms=True, forecast_recomputed=False,
        counterfactual_outcomes_evaluated=False, controller_changed=False, windows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-name', required=True)
    parser.add_argument('--start-frame', type=int, default=2000)
    parser.add_argument('--end-frame', type=int, default=4000)
    args = parser.parse_args(); root = BASE/args.root_name
    report = evaluate(root, start_frame=args.start_frame, end_frame=args.end_frame)
    with (root/f'saved_waypoint_alignment_{args.start_frame}_{args.end_frame}_v1.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != 'windows'}))
