"""Compare unfitted command integration with learned forecasts on identical windows."""
import argparse
from collections import defaultdict
import json
import numpy as np
from lewm.commanded_planar_motion_development import forecast
from lewm.geometry_progress_pilot_development import ACTIONS
from scripts.compare_continuous_navigation_arms_development import path, read


def metrics(rows):
    if not rows:return dict(windows=0)
    return dict(windows=len(rows),**{name+'_endpoint_xy_rmse_m':
        float(np.sqrt(np.mean([r[name+'_endpoint_error_m']**2 for r in rows])))
        for name in ('command','raw','corrected')},
        corrected_better_than_command_windows=sum(r['corrected_endpoint_error_m']<r['command_endpoint_error_m'] for r in rows),
        command_endpoint_error_maximum_m=max(r['command_endpoint_error_m'] for r in rows))


def evaluate(root):
    previous=read(root,'saved_executed_motion_forecast_evaluation_v1.json')
    plans={p['frame']:p for p in read(root,'planning.json') if 'selection' in p}
    rows=[];groups=defaultdict(list)
    for original in previous['rows']:
        p=plans[original['frame']]
        if p['action']!=original['action']:raise ValueError('same executed candidate required')
        prediction=forecast(p['committed_prefix'],pulse=bool(p['motion_correction'].get('terminal_translation_pulse',False)))
        endpoint=prediction[ACTIONS.index(p['action']),6,:2]
        error=float(np.linalg.norm(endpoint-original['actual_endpoint_xy_m']))
        row=original|dict(command_endpoint_xy_m=endpoint.tolist(),command_endpoint_error_m=error)
        rows.append(row);groups[row['group']].append(row)
    return dict(root_name=root.name,**metrics(rows),by_action_group={g:metrics(r) for g,r in groups.items()},
        horizon_ns=700_000_000,identical_executed_windows=True,command_model_fitted=False,
        command_forecast_uses_pose_or_native_state=False,actual_endpoint_evaluator_only=True,
        contact_prediction_compared=False,closed_loop_control_intervention=False,
        trajectory_conditional=True,overlapping_windows=True,rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root-name',required=True)
    args=parser.parse_args();root=path(args.root_name);result=evaluate(root)
    with (root/'command_motion_forecast_comparison_v1.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
