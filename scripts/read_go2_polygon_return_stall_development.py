"""Read recorded return decisions without replaying alternative controllers."""
from collections import Counter
import json
import math

import numpy as np

from scripts import run_go2_polygon_floor_repeatability_development as run


def main():
    output = run.BASE / run.root_name(4) / 'return_stall_readout_v1.json'
    if output.exists():
        raise ValueError('preserve completed diagnostic')
    rows = []
    for number in range(1, 5):
        root = run.BASE / run.root_name(number)
        read = lambda name: json.loads((root / name).read_text())
        result = read('frozen_readout_navigation_readout_v1.json')
        goal = next(a for a in result['navigation']['arrivals'] if a['phase'] == 'OUTBOUND')
        goal_frame = goal['frame']
        plans = [r for r in read('planning.json') if 'selection' in r and r['frame'] > goal_frame]
        poses = [r['registered_pose'] for r in read('poses.json')
            if r['registered_pose'] is not None and r['frame'] >= goal_frame]
        position = np.asarray([p['position_initial_body_m'][:2] for p in poses])
        yaw = np.unwrap([math.atan2(p['rotation_initial_body_from_current_body'][1][0],
            p['rotation_initial_body_from_current_body'][0][0]) for p in poses])
        requests = [r for r in read('requests.json') if r.get('mission_phase') == 'RETURN']
        angular = [math.copysign(1., r['applied_command'][2]) for r in requests if r['applied_command'][2]]
        reasons = Counter(r['reason'] for r in requests)
        events = Counter()
        latched_preceding_recovery = 0
        previous = None
        transitions = []
        preferred_turns = Counter()
        for plan in plans:
            selection = plan['selection']; latch = selection.get('clearance_turn') or {}
            if latch.get('event'):
                events[latch['event']] += 1
            if (previous is not None and (previous['selection'].get('clearance_turn') or {}).get('active')
                    and previous['route_status'] != 'LOW_VISUAL_SUPPORT_REQUIRES_MEASURED_VIEW'
                    and plan['route_status'] == 'LOW_VISUAL_SUPPORT_REQUIRES_MEASURED_VIEW'):
                latched_preceding_recovery += 1
                transitions.append(dict(frame=plan['frame'], previous_frame=previous['frame'],
                    previous_latch=previous['selection']['clearance_turn'],
                    next_latch=latch, previous_action=previous['action'], action=plan['action']))
            preferred = selection.get('before_memory_filter_action')
            by_action = {c['action']: c for c in selection['memory_forecast_candidates']}
            if preferred in ('left_turn', 'right_turn'):
                preferred_turns['total'] += 1
                candidate = by_action[preferred]
                preferred_turns['reserve_blocked'] += not candidate['nominal_predicted_path_clear']
                preferred_turns['nominal_footprint_clear_but_reserve_blocked'] += (
                    candidate['nominal_footprint_path_clear'] and not candidate['nominal_predicted_path_clear'])
            previous = plan
        rows.append(dict(assignment=number, arm=result['arm'], goal_frame=goal_frame,
            round_trip=result['navigation']['round_trip'],
            return_plans=len(plans), actions=dict(Counter(p['action'] for p in plans)),
            route_statuses=dict(Counter(p['route_status'] for p in plans)),
            return_request_samples=len(requests), applied_translation_samples=sum(
                bool(r['applied_command'][0] or r['applied_command'][1]) for r in requests),
            applied_turn_sign_reversals=sum(a != b for a, b in zip(angular, angular[1:])),
            request_reasons=dict(reasons), clearance_turn_events=dict(events),
            preferred_turns=dict(preferred_turns),
            active_route_latch_followed_by_visual_recovery=latched_preceding_recovery,
            latch_to_recovery_transitions=transitions,
            observed_position_max_displacement_from_arrival_m=float(np.linalg.norm(position-position[0], axis=1).max()),
            observed_return_yaw_span_rad=float(np.ptp(yaw)),
            accepted_return_pose_frames=len(poses)))
        print('RETURN_STALL_READOUT',number,rows[-1]['actions'],
            'route_latches_interrupted',latched_preceding_recovery,flush=True)
    output.write_text(json.dumps(dict(rows=rows, recorded_decisions_only=True,
        native_state_or_maze_geometry_used=False, alternative_navigation_outcome_proven=False,
        feature_counts_are_not_calibrated_confidence=True),indent=2)+'\n')


if __name__ == '__main__':
    main()
