"""Describe recorded planner choices and dispatch vetoes after a native run."""
from collections import Counter
import json
import numpy as np


def summarize_decisions(root):
    read = lambda name:json.loads((root/name).read_text())
    plans = [p for p in read('planning.json') if 'selection' in p]
    requests = read('requests.json')
    with np.load(root/'native/physics_trace.npz',allow_pickle=False) as trace:
        physical_end = int(round(float(trace['timestamp_s'][-1])*1e9))
    longest = dict(duration_s=0., started_ns=None, ended_ns=None)
    start = None
    for i, row in enumerate(requests):
        now = row['simulator_ns']
        end = min(physical_end, requests[i+1]['simulator_ns'] if i+1<len(requests) else physical_end)
        if not any(row['requested_command']):
            if start is None:
                start = now
            if (end-start)/1e9>longest['duration_s']:
                longest = dict(duration_s=(end-start)/1e9, started_ns=start, ended_ns=end)
        else:
            start = None
    decisions = []
    for p in plans:
        selected = p['selection']
        rows = selected.get('scan_utilities',selected.get('candidates',[]))
        ranked = [r for r in rows if 'utility_m' in r]
        highest = max(ranked,key=lambda r:r['utility_m'])['action'] if ranked else None
        memory = selected.get('memory_forecast_candidates',[])
        stopping = selected.get('planned_stopping_projection',{})
        decisions.append(dict(frame=p['frame'], action=p['action'], route_status=p['route_status'],
            committed=p.get('committed'), on_time=p.get('on_time'),
            highest_recorded_utility_action=highest,
            before_memory_filter_action=selected.get('before_memory_filter_action'),
            memory_forecast_status=selected.get('memory_forecast_status'),
            candidates_with_nominal_predicted_clearance=[r['action'] for r in memory if r.get('nominal_footprint_path_clear')],
            candidates_with_full_predicted_reserve=[r['action'] for r in memory if r.get('full_reserve_path_clear')],
            stopping_projection_changed_action=stopping.get('changed',False),
            stopping_projection_before_action=stopping.get('before_action'),
            waypoint_body_xy_m=selected.get('waypoint_body_xy_m')))
    hold = [r for r in decisions if r['action']=='hold']
    return dict(selected_plans=len(plans),
        selected_actions=dict(Counter(r['action'] for r in decisions)),
        route_statuses=dict(Counter(r['route_status'] for r in decisions)),
        selected_hold_plans=len(hold),
        hold_with_highest_recorded_utility_also_hold=sum(r['highest_recorded_utility_action']=='hold' for r in hold),
        hold_with_higher_recorded_nonhold_utility=sum(r['highest_recorded_utility_action'] not in (None,'hold') for r in hold),
        stopping_projection_changes=sum(r['stopping_projection_changed_action'] for r in decisions),
        memory_forecast_statuses=dict(Counter(str(r['memory_forecast_status']) for r in decisions)),
        requested_command_categories=dict(Counter('translation' if any(r['requested_command'][:2])
            else 'turn' if r['requested_command'][2] else 'hold' for r in requests)),
        zero_request_reasons=dict(Counter(r['reason'] for r in requests if not any(r['requested_command']))),
        nonzero_request_reasons=dict(Counter(r['reason'] for r in requests if any(r['requested_command']))),
        longest_consecutive_zero_request=longest, rows=decisions,
        scope='post hoc recorded choices and gates; highest recorded utility is not a counterfactual navigation result or an isolated causal attribution')
