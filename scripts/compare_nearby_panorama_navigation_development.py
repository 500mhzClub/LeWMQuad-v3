"""Compare physical outcomes and measured frontier-survey use on one layout."""
import argparse
from collections import Counter
import json

import numpy as np

from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.terminal_translation_pulse_development import command_sequences
from scripts.compare_continuous_navigation_arms_development import path, read, summarize


def diagnose(root):
    plans = [p for p in read(root, 'planning.json') if 'selection' in p]
    visits = read(root, 'frontier_visits.json')
    frames = {r['frame']: r for r in read(root, 'native/in_memory_camera_observations.json')['frames']}
    requests = {r['simulator_ns']: r['requested_command'] for r in read(root, 'requests.json')}
    directed = [p for p in plans if p.get('frontier_visit', {}).get('nearby_completed_panorama')]
    executed = []
    for p in directed:
        commands = command_sequences(p['committed_prefix'],
            pulse=bool(p['motion_correction'].get('terminal_translation_pulse', False)))[ACTIONS.index(p['action']), :7]
        times = [p['measured_ns']+j*100_000_000+k*20_000_000 for j in range(7) for k in range(5)]
        matched = all(t in requests for t in times) and np.allclose(
            np.array([requests[t] for t in times]), np.repeat(commands, 5, axis=0), rtol=0, atol=1e-8)
        executed.append(dict(frame=p['frame'], action=p['action'], on_time=p['on_time'],
            requested_sequence_through_700ms_matched=bool(matched)))
    events = []
    for v in visits['events']:
        witness = v.get('nearby_completed_panorama')
        events.append(dict(started_ns=v['started_ns'], completed_ns=v['completed_ns'],
            seconds=(v['completed_ns']-v['started_ns'])/1e9,
            full_panorama=bool(v.get('panoramic_frontier_view')),
            directed_revisit=witness is not None,
            target_xy_m=v['target_xy_m'], view_start_map_xy_m=v['view_start_map_xy_m'],
            completed_view_stages=len(v.get('completed_view_stages', [])),
            post_alignment_map_measured=frames[v['map_frame']]['measured_ns']>=v['aligned_ns'],
            nearby_completed_panorama=witness))
    pending_seconds = (max(f['measured_ns'] for f in frames.values())-visits['pending']['started_ns'])/1e9 if visits['pending'] else 0.
    return dict(plans=len(plans), actions=dict(Counter(p['action'] for p in plans)),
        route_status_counts=dict(Counter(p['route_status'] for p in plans)),
        completed_frontier_surveys=len(events),
        completed_full_panoramas=sum(e['full_panorama'] for e in events),
        completed_directed_revisits=sum(e['directed_revisit'] for e in events),
        completed_frontier_survey_seconds=sum(e['seconds'] for e in events),
        pending_frontier_survey_elapsed_seconds=pending_seconds,
        total_frontier_survey_elapsed_seconds=sum(e['seconds'] for e in events)+pending_seconds,
        pending_frontier_visit=visits['pending'],
        directed_revisit_plan_count=len(directed),
        directed_revisit_plans_on_time=sum(p['on_time'] for p in directed),
        directed_revisit_plans_requested_sequence_matched=sum(e['requested_sequence_through_700ms_matched'] for e in executed),
        completed_events=events, directed_plan_execution=executed,
        counterfactual_time_savings_established=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=(0, 1), required=True)
    args = parser.parse_args(); i = args.layout_index
    output = path(f'go2_nearby_panorama_comparison_layout{i:02d}_v1_attempt_001')
    if output.exists():
        raise ValueError('preserve previous comparison')
    roots = {c: path(f'go2_nearby_panorama_{c}_native_layout{i:02d}_4800_v1_attempt_001')
        for c in ('baseline', 'directed')}
    launches = {c: read(r, 'launch.json') for c, r in roots.items()}
    treatment = {'owner', 'comparison_condition', 'nearby_panorama_directed_view'}
    a, b = launches['baseline'], launches['directed']
    changed = sorted(k for k in a.keys()|b.keys() if a.get(k)!=b.get(k))
    if set(changed)-treatment:
        raise ValueError(f'non-treatment settings or sources differ: {changed}')
    summaries = {c: summarize(r) for c, r in roots.items()}
    diagnostics = {c: diagnose(r) for c, r in roots.items()}
    requests = {c: {p['simulator_ns']:p['requested_command'] for p in read(r, 'requests.json')}
        for c, r in roots.items()}
    first_difference = next((t for t in sorted(requests['baseline'].keys()&requests['directed'].keys())
        if requests['baseline'][t]!=requests['directed'][t]), None)
    directed_starts = [p['frontier_visit']['started_ns'] for p in read(roots['directed'], 'planning.json')
        if p.get('frontier_visit', {}).get('nearby_completed_panorama')]
    first_directed = min(directed_starts, default=None)
    report = dict(layout_index=i, comparison='nearby_completed_panorama_directed_revisits',
        conditions=summaries, mechanism=diagnostics, changed_launch_fields=changed,
        common_settings_and_sources_equal=True,
        first_requested_command_difference_ns=first_difference,
        first_directed_revisit_started_ns=first_directed,
        execution_diverged_before_first_directed_revisit=(first_difference<first_directed
            if first_difference is not None and first_directed is not None else None),
        causal_episode_improvement_established=False, repeatability_established=False,
        native_state_evaluator_only=True, host_real_time_qualified=False)
    output.mkdir()
    with (output/'result.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({c:dict(arrivals=s['independent_arrival_evaluation'],
        frontier_surveys=diagnostics[c]['completed_frontier_surveys'],
        frontier_survey_seconds=diagnostics[c]['total_frontier_survey_elapsed_seconds']) for c,s in summaries.items()}))


if __name__ == '__main__':
    main()
