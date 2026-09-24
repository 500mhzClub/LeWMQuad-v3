"""Read-only post-hoc transfer diagnostic; no fitted model enters a controller.

Fit six pulse+brake response means from the older two nominal command-collection
episodes. Predict room responses from requested action/duration only. No actual
future yaw/pose, native state or room response enters the fitted lookup table.
This tiny descriptive model is not JEPA, independent validation or qualification.
"""
import json
import math
import numpy as np
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest
from scripts.run_go2_room_return_pulse_v1 import OUTPUT,TRIALS
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings


def wrap(x):return np.arctan2(np.sin(x),np.cos(x))


def fitting_table():
    old=ROOT/'.generated/go2_command_pulse_response_v1_attempt_001'
    if digest(old/'raw_pulse_audit.json')!='af809e5cb6101bf67169e1b2a3f5e66be424b53c316d7f4e40854ee9fd5d1d46':
        raise ValueError('fixed older pulse audit required')
    audit=read_json(old,'raw_pulse_audit.json');samples={}
    for c in ('nominal_a','nominal_b'):
        name=c+'_pulse_evaluation.json'
        if digest(old/name)!=audit['evaluation_sha256'][c]:raise ValueError('fixed old evaluation required')
        for event in read_json(old,name)['events']:
            if event['command_name'] not in ('forward_bank','left_bank','right_bank'):continue
            if not event['pulse_complete']:raise ValueError('complete fitting pulse required')
            v=event['endpoints']['brake_20']['visual']
            key=(tuple(event['requested_command']),event['pulse_ticks'])
            samples.setdefault(key,[]).append([*v['displacement_body_m'],v['yaw_change_rad']])
    table={key:np.mean(values,axis=0) for key,values in samples.items()}
    assert len(table)==6 and all(len(v)==2 for v in samples.values())
    return table,digest(old/'raw_pulse_audit.json')


def report():
    table,source_audit_hash=fitting_table()
    result=read_json(OUTPUT,'result.json')
    verify_bindings({str((OUTPUT/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()})
    reports={}
    for c in TRIALS:
        rows=read_json(OUTPUT/c,'servo_decisions.json');actions={};groups={'forward':[],'turn':[]}
        for row in rows:
            local=row['decision']['execution']['local_decision']
            if local is None:continue
            d=local['diagnostic'];anchor=local['goal']['anchor_ns']
            if 'new_pulse' in d:
                pulse=d['new_pulse'];actions[(anchor,pulse['pulse_index'])]=(tuple(local['requested_command']),pulse['ticks'])
            if 'completed_action_response' in d:
                response=d['completed_action_response'];key=actions[(anchor,response['pulse_index'])]
                observed=np.array([*response['displacement_start_body_m'],response['yaw_change_rad']])
                groups[response['kind']].append((table[key],observed))
        reports[c]={}
        for kind,pairs in groups.items():
            if not pairs:reports[c][kind]=dict(completed_events=0);continue
            pred,actual=np.array([p[0] for p in pairs]),np.array([p[1] for p in pairs])
            reports[c][kind]=dict(completed_events=len(pairs),
                mean_table_planar_rmse_m=float(np.sqrt(np.mean(np.sum((pred[:,:2]-actual[:,:2])**2,axis=1)))),
                zero_translation_planar_rmse_m=float(np.sqrt(np.mean(np.sum(actual[:,:2]**2,axis=1)))),
                mean_table_yaw_rmse_rad=float(np.sqrt(np.mean(wrap(pred[:,3]-actual[:,3])**2))),
                zero_rotation_yaw_rmse_rad=float(np.sqrt(np.mean(wrap(actual[:,3])**2))))
    return dict(scope='post-hoc descriptive pulse-transfer diagnostic; no controller use or JEPA',
        fitting_episodes=['nominal_a','nominal_b'],samples_per_cell=2,
        endpoint_caveat='old fitting endpoints use20brake ticks; room responses may wait longer for quiet; state/history and horizon mismatch remain',
        source_audit_sha256=source_audit_hash,room_result_sha256=digest(OUTPUT/'result.json'),
        table=[dict(command=list(k[0]),pulse_ticks=k[1],mean_sensor_delta_xyz_yaw=v.tolist()) for k,v in table.items()],
        conditions=reports)


def main():
    print(json.dumps(report(),indent=2),flush=True)


if __name__=='__main__':main()
