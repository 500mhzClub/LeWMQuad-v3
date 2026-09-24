#!/usr/bin/env python3
"""Read-only reconciliation of the superseded development screen semantics."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
src=ROOT/'.generated/minimal_spatial_topological_planning_spike_v1/result.json'
out=ROOT/'.generated/safe_local_waypoint_purpose_built_v1/old_metric_reconciliation.json'
r=json.loads(src.read_text())
t=r['true_future']
states=[]
for st in t['per_state']:
    states.append({'state_id':st['state'],'selected_candidate':st['selected'],
                   'selected_progress':st['progress'],'best_progress':st['oracle_best'],
                   'worst_progress':None,'normalized_regret':st['regret'],
                   'selected_safety':st['unsafe'],'safe_candidates_available':'not stored in old result',
                   'safe_positive_progress_candidates':'not stored in old result'})
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'source':str(src),'selected_unsafe_rate':{'value':t['selected_unsafe_rate'],'denominator':'number of selected held-out states (2)','semantics':'selected branch safety only'},'normalized_regret':{'formula':'mean absolute regret / (mean absolute progress + 1e-6)','bounded_0_1':False,'note':'superseded screen used best over all candidates, not best safe'},'per_state':states},indent=2))
print(out)
