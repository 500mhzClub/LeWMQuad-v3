"""Diagnose the exposed view-arc trial without asserting unexecuted outcomes."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

BASE = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
ROOT = 'go2_measured_view_arc_recovery_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001'


def main():
    root = BASE/ROOT
    read = lambda name: json.loads((root/name).read_text())
    plans = [r for r in read('planning.json') if 'selection' in r]
    poses = {r['frame']: r['registered_pose'] for r in read('poses.json')}
    requests = read('requests.json')
    dispatched = defaultdict(list)
    for r in requests:
        if 'command_observation_ns' in r:
            dispatched[r['command_observation_ns']].append(r)
    stalled = [r for r in plans if 800 <= r['frame'] < 3800]
    releases = []
    for r in stalled:
        receipt = r['selection'].get('full_reserve_heading_release')
        if not receipt:
            continue
        relatch = next((s for s in stalled if s['frame'] > r['frame'] and
            s['selection'].get('clearance_turn', {}).get('event') == 'CLEAR_ALTERNATIVE_TURN_LATCHED'), None)
        applied = [s for s in dispatched[r['measured_ns']] if
            np.allclose(s['applied_command'], r['selection']['requested_command'], atol=1e-7)]
        releases.append(dict(frame=r['frame'], on_time=r['on_time'],
            previous_action=receipt['previous_action'], selected_action=r['action'],
            matching_applied_intervals=len(applied),
            next_relatch_frame=None if relatch is None else relatch['frame'],
            seconds_to_next_relatch=None if relatch is None else
                (relatch['measured_ns']-r['measured_ns'])/1e9))
    positions = np.asarray([poses[r['frame']]['position_initial_body_m'][:2] for r in stalled])
    deltas = [r['seconds_to_next_relatch'] for r in releases if r['seconds_to_next_relatch'] is not None]
    turns = [r for r in stalled if r['action'] in ('left_turn', 'right_turn')]
    translations = ('forward', 'left_arc', 'right_arc')
    eligible = lambda r: any(c['nominal_predicted_path_clear'] and next(
        s['projection_clear'] for s in r['selection']['planned_stopping_projection']['candidates']
        if s['action'] == c['action']) for c in r['selection']['memory_forecast_candidates']
        if c['action'] in translations)
    result = dict(schema='measured_view_arc_trial_diagnosis.v1',
        evaluation=read('short_pulse_navigation_evaluation_v1.json'),
        fallback_selections=sum('measured_view_arc_recovery' in r['selection'] for r in plans),
        fallback_efficacy_tested=False,
        visual_recovery_plans=sum(r['route_status']=='LOW_VISUAL_SUPPORT_REQUIRES_MEASURED_VIEW' for r in plans),
        visual_recovery_publications=len(read('visual_dispatch_events.json')),
        diagnosed_interval_frames=[800, 3800], interval_end_exclusive=True,
        interval_plans=len(stalled), interval_plans_on_time=sum(r['on_time'] for r in stalled),
        interval_actions=dict(Counter(r['action'] for r in stalled)),
        interval_routes=dict(Counter(r['route_status'] for r in stalled)),
        interval_registered_xy_span_m=np.ptp(positions, axis=0).tolist(),
        interval_registered_endpoint_displacement_m=float(np.linalg.norm(positions[-1]-positions[0])),
        consecutive_turn_direction_reversals=sum(a['action']!=b['action'] for a,b in zip(turns, turns[1:])),
        reversal_count_ignores_intervening_nonturn_plans=True,
        plans_with_clear_translation_and_stopping=sum(eligible(r) for r in stalled),
        early_heading_releases=len(releases),
        early_heading_releases_with_applied_commands=sum(r['matching_applied_intervals']>0 for r in releases),
        releases_followed_by_relatch_within_2s=sum(d<=2 for d in deltas),
        median_seconds_to_next_relatch=float(np.median(deltas)) if deltas else None,
        release_rows=releases,
        release_causally_prevents_completion_proven=False,
        no_release_counterfactual_navigation_tested=False,
        full_failure_depth_preserved=True)
    with (root/'turn_cycle_diagnosis_v1.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='release_rows'}, indent=2))


if __name__ == '__main__':
    main()
