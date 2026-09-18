"""Compare recorded neural yaw with ideal command yaw on executed windows."""
import argparse
import json
import numpy as np
from lewm.commanded_planar_motion_development import forecast
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.physical_execution_development import rotation_xyzw
from scripts.compare_continuous_navigation_arms_development import path,read


def wrapped(angle):
    return float(np.arctan2(np.sin(angle),np.cos(angle)))


def metrics(rows):
    result=dict(windows=len(rows))
    for interval in ('endpoint','commit'):
        for source in ('neural','command','applied'):
            errors=np.asarray([r[f'{source}_{interval}_error_rad'] for r in rows])
            result[f'{source}_{interval}_rmse_deg']=float(np.degrees(np.sqrt(np.mean(errors**2)))) if len(errors) else None
    return result


def evaluate(root):
    applied_source=read(root,'launch.json').get('forecast_yaw_source','learned')
    plans={p['frame']:p for p in read(root,'planning.json') if 'selection' in p}
    frames={f['frame']:f for f in read(root,'native/in_memory_camera_observations.json')['frames']}
    with np.load(root/'native/physics_trace.npz',allow_pickle=False) as arrays:
        physics=arrays['base_pose_world'].copy()
    rows=[]
    for window in read(root,'saved_executed_motion_forecast_evaluation_v1.json')['rows']:
        frame=window['frame'];p=plans[frame];c=p['motion_correction'];index=ACTIONS.index(p['action'])
        commanded=forecast(p['committed_prefix'],pulse=bool(c['terminal_translation_pulse']))
        if 'applied_prediction_after_yaw_ablation' in c:
            neural=np.asarray(c['upstream_prediction_for_yaw_ablation'])
            applied=np.asarray(c['applied_prediction_after_yaw_ablation'])
            if c['forecast_yaw_source']!=applied_source or applied_source not in ('learned','command'):
                raise ValueError('same assigned and actually applied yaw source required')
            expected=neural[:,:,2:4] if applied_source=='learned' else commanded[:,:,2:4]
            if (neural.shape!=(6,8,5) or applied.shape!=(6,8,5)
                    or not np.array_equal(applied[:,:,[0,1,4]],neural[:,:,[0,1,4]])
                    or not np.allclose(applied[:,:,2:4],expected,rtol=0,atol=1e-7)):
                raise ValueError('actual final yaw intervention differs')
        else:
            neural=np.asarray(c['upstream_prediction_for_contact_ablation'])
            applied=np.asarray(c['applied_prediction_after_contact_ablation'])
            if applied.shape!=(6,8,5) or not np.array_equal(applied[:,:,2:4],neural[:,:,2:4]):
                raise ValueError('unchanged recorded neural yaw required')
        headings=[]
        for f in (frame,frame+3,frame+7):
            rotation=rotation_xyzw(physics[frames[f]['physical_sample_index'],3:])
            headings.append(float(np.arctan2(rotation[1,0],rotation[0,0])))
        actual_endpoint=wrapped(headings[2]-headings[0])
        actual_commit=wrapped(headings[2]-headings[1])
        row=dict(frame=frame,action=p['action'],group=window['group'],
            actual_endpoint_yaw_rad=actual_endpoint,actual_commit_yaw_rad=actual_commit)
        for source,prediction in (('neural',neural),('command',commanded),('applied',applied)):
            start=float(np.arctan2(prediction[index,2,2],prediction[index,2,3]))
            end=float(np.arctan2(prediction[index,6,2],prediction[index,6,3]))
            row[f'{source}_endpoint_error_rad']=wrapped(end-actual_endpoint)
            row[f'{source}_commit_error_rad']=wrapped(wrapped(end-start)-actual_commit)
        rows.append(row)
    return dict(root_name=root.name,**metrics(rows),
        actual_applied_yaw_source=applied_source,
        neural_is_saved_alternative_when_command_applied=applied_source=='command',
        by_action={a:metrics([r for r in rows if r['action']==a]) for a in ACTIONS},
        matched_requested_sequence_through_ns=700_000_000,
        commit_evaluation_interval_ns=[300_000_000,700_000_000],
        actual_yaw_definition='wrapped_difference_of_world_heading_matching_training_targets',
        target_definition_sources=['lewm/causal_subtrajectory_development.py','lewm/moving_prefix_evidence_development.py'],
        native_state_evaluator_only=True,neural_forecast_recomputed=False,
        command_predictor_models_gait_inertia_or_slip=False,
        overlapping_windows_not_independent=True,alternative_navigation_outcome_established=False,
        rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root-name',required=True)
    root=path(parser.parse_args().root_name);report=evaluate(root)
    with (root/'saved_neural_command_yaw_evaluation_v1.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
