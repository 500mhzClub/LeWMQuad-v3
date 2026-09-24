"""Retrospective same-window XY evaluation of the frozen pose/command control."""
import argparse
from collections import defaultdict
import hashlib
import json
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.terminal_translation_pulse_development import command_sequences
from scripts.fit_closed_loop_motion_residual_development import features, pose_features
from scripts.fit_pose_command_motion_control_development import OUTPUT, COLUMNS
from scripts.compare_continuous_navigation_arms_development import path, read


def summarize(rows):
    if not rows:return dict(windows=0)
    return dict(windows=len(rows),**{name+'_endpoint_xy_rmse_m':float(np.sqrt(np.mean([
        r[name+'_endpoint_error_m']**2 for r in rows]))) for name in ('pose_command','corrected')},
        pose_command_endpoint_error_maximum_m=max(r['pose_command_endpoint_error_m'] for r in rows))


def evaluate(root):
    frozen=read(OUTPUT,'result.json');fit_file=OUTPUT/'motion_fit.npz'
    if hashlib.sha256(fit_file.read_bytes()).hexdigest()!=frozen['motion_fit_sha256']:
        raise ValueError('fixed pose-command fit identity required')
    with np.load(fit_file,allow_pickle=False) as arrays:fit={k:arrays[k].copy() for k in arrays.files}
    poses={r['frame']:r['registered_pose'] for r in read(root,'poses.json')}
    plans={r['frame']:r for r in read(root,'planning.json') if 'selection' in r}
    original=read(root,'saved_executed_motion_forecast_evaluation_v1.json')
    rows=[];groups=defaultdict(list)
    for row in original['rows']:
        frame=row['frame'];p=plans[frame]
        history={f:poses[f] for f in range(frame-3,frame+1)}
        past,_,_=pose_features(history,frame)
        commands=command_sequences(p['committed_prefix'],pulse=bool(p['motion_correction'].get('terminal_translation_pulse',False)))[ACTIONS.index(p['action'])]
        # Neural feature slots are discarded; no learned outcome enters x.
        x=features(np.zeros((8,5)),past,commands)[:,COLUMNS]
        h=6;endpoint=((x[h]-fit['mean'][h])/fit['scale'][h])@fit['coefficient'][h]+fit['bias'][h]
        error=float(np.linalg.norm(endpoint-row['actual_endpoint_xy_m']))
        item=dict(frame=frame,action=p['action'],group=row['group'],
            pose_command_endpoint_xy_m=endpoint.tolist(),pose_command_endpoint_error_m=error,
            corrected_endpoint_error_m=row['corrected_endpoint_error_m'])
        rows.append(item);groups[item['group']].append(item)
    return dict(root_name=root.name,fit_root=OUTPUT.name,fit_sha256=frozen['motion_fit_sha256'],
        **summarize(rows),by_action_group={g:summarize(r) for g,r in groups.items()},
        horizon_ns=700_000_000,identical_executed_windows=True,
        neural_forecast_inputs_used=False,only_four_causal_visual_poses_and_known_commands=True,
        future_pose_used_as_input=False,actual_endpoint_evaluator_only=True,
        retrospective_development_comparison=True,closed_loop_control_intervention=False,
        yaw_and_contact_not_compared=True,overlapping_windows=True,rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root-name',required=True)
    args=parser.parse_args();root=path(args.root_name);result=evaluate(root)
    with (root/'pose_command_motion_control_evaluation_v1.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
