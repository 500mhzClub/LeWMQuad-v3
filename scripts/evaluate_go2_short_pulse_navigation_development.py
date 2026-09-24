"""Physical outcomes and recorded actual forecast treatment, including failures."""
import argparse
import json
import numpy as np
from lewm.commanded_planar_motion_development import forecast as nominal
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.physical_execution_development import rotation_xyzw
from scripts import run_go2_short_pulse_navigation_development as study
from scripts.evaluate_go2_stopping_projection_transfer_development import save_or_read,arrivals,summarize,xy


def read(root,name):return json.loads((root/name).read_text())


def yaw_metrics(root,windows):
    plans={p['frame']:p for p in read(root,'planning.json') if 'selection' in p}
    frames={p['frame']:p for p in read(root,'native/in_memory_camera_observations.json')['frames']}
    with np.load(root/'native/physics_trace.npz',allow_pickle=False) as arrays:poses=arrays['base_pose_world']
    rows=[]
    for w in windows['rows']:
        frame=w['frame'];p=plans[frame];c=p['motion_correction'];i=ACTIONS.index(w['action'])
        endpoints=[rotation_xyzw(poses[frames[f]['physical_sample_index'],3:]) for f in (frame,frame+7)]
        actual=np.arctan2(endpoints[1][1,0],endpoints[1][0,0])-np.arctan2(endpoints[0][1,0],endpoints[0][0,0])
        predictions=dict(applied=np.asarray(c['applied_prediction_after_yaw_ablation']),
            neural=np.asarray(c['upstream_prediction_for_yaw_ablation']),
            nominal=nominal(p['committed_prefix'],pulse=bool(c['terminal_translation_pulse'])))
        errors={}
        for name,prediction in predictions.items():
            delta=np.arctan2(prediction[i,6,2],prediction[i,6,3])-actual
            errors[name]=float(np.arctan2(np.sin(delta),np.cos(delta)))
        delta=np.asarray(c['command_history_forecast_xy_yaw'])[i,6,2]-actual
        errors['command_history']=float(np.arctan2(np.sin(delta),np.cos(delta)))
        rows.append(dict(frame=frame,group=w['group'],errors_rad=errors))
    return dict(windows=len(rows),rmse_deg={name:float(np.degrees(np.sqrt(np.mean([r['errors_rad'][name]**2 for r in rows]))))
        if rows else None for name in ('applied','neural','nominal','command_history')},rows=rows,
        yaw_truth='wrapped world-heading difference',native_state_evaluator_only=True,
        matched_horizon_ms=700,overlapping_windows=True)


def same_window_xy(root,windows):
    """Compare recorded forecasts on the executed action, not alternative policies."""
    plans={p['frame']:p for p in read(root,'planning.json') if 'selection' in p}
    rows=[]
    for w in windows['rows']:
        p=plans[w['frame']];c=p['motion_correction'];i=ACTIONS.index(w['action'])
        actual=np.asarray(w['actual_endpoint_xy_m'])
        predictions=dict(applied=c['corrected_forecast_xy_m'],neural=c['raw_forecast_xy_m'],
            pose_command=c['pose_command_forecast_xy_m'],command_history=c['command_history_forecast_xy_yaw'],
            nominal=nominal(p['committed_prefix'],pulse=bool(c['terminal_translation_pulse'])))
        rows.append(dict(frame=w['frame'],group=w['group'],errors_m={
            name:float(np.linalg.norm(np.asarray(pred)[i,6,:2]-actual)) for name,pred in predictions.items()}))
    def metrics(population):
        return dict(windows=len(population),rmse_mm={
            name:float(1000*np.sqrt(np.mean([r['errors_m'][name]**2 for r in population])))
            for name in ('applied','neural','pose_command','command_history','nominal')}) if population else dict(windows=0)
    return dict(**metrics(rows),by_action_group={g:metrics([r for r in rows if r['group']==g])
        for g in sorted({r['group'] for r in rows})},rows=rows,matched_horizon_ms=700,
        native_state_evaluator_only=True,overlapping_windows=True,
        alternative_navigation_outcomes_evaluated=False)


def evaluate(number):
    index,arm=study.ASSIGNMENTS[number-1];root=study.BASE/study.ROOT.format(index=index,arm=arm)
    launch=read(root,'launch.json');fixed=json.loads(study.PLAN.read_text())
    if launch['study_arm']!=arm or launch['layout_index']!=index:raise ValueError('fixed assignment required')
    if not (root/'result.json').exists() and not (root/'failure.json').exists():raise ValueError('terminal owner required')
    if not (root/'native/in_memory_camera_observations.json').exists():
        raise ValueError('wait for native recording persistence and owner exit before evaluating')
    plans=[p for p in read(root,'planning.json') if 'selection' in p]
    expected=arm if arm in ('pose_command','command_history') else 'neural'
    for p in plans:
        selected=p['selection']
        if arm=='reactive':
            if 'motion_correction' in p or selected.get('candidate_future_outcomes_evaluated') is not False:
                raise ValueError('reactive selection consumed forecasts')
            continue
        c=p['motion_correction'];applied=np.asarray(c['applied_prediction_after_yaw_ablation'])
        neural=np.asarray(c['upstream_prediction_for_yaw_ablation'])
        if c['prediction_source']!=expected or c['external_neural_correction_applied']:
            raise ValueError('actual assigned raw model treatment required')
        if (applied.shape!=(6,8,5) or not np.isfinite(applied).all() or not np.all(applied[:,:,4]==-1000.)
                or not np.array_equal(applied[:,:,:2],c['corrected_forecast_xy_m'])
                or c['command_history_fit_sha256']!=fixed['command_history_fit_sha256']):
            raise ValueError('complete recorded assigned predictions required')
        if expected=='neural':
            np.testing.assert_array_equal(applied[:,:,:4],neural[:,:,:4])
        elif expected=='pose_command':
            # Runtime assigns the double-precision fit into the model's
            # float32 output buffer. Compare that exact conversion.
            np.testing.assert_array_equal(applied[:,:,:2],np.asarray(c['pose_command_forecast_xy_m'],dtype=np.float32))
            n=nominal(p['committed_prefix'],pulse=bool(c['terminal_translation_pulse']))
            np.testing.assert_allclose(applied[:,:,2:4],n[:,:,2:4],rtol=0,atol=1e-7)
        else:
            cmd=np.asarray(c['command_history_forecast_xy_yaw'])
            np.testing.assert_allclose(applied[:,:,:2],cmd[:,:,:2],rtol=0,atol=1e-7)
            np.testing.assert_allclose(applied[:,:,2:4],np.stack((np.sin(cmd[:,:,2]),np.cos(cmd[:,:,2])),axis=-1),rtol=0,atol=1e-7)
        check=selected['model_input_treatment']
        if check['input_variant']!='full' or check['checked_forward_calls']!=1:
            raise ValueError('one actual full-input model call per plan required')
        if ('instantaneous_ranking' in selected)!=(arm=='instantaneous') or 'planned_stopping_projection' not in selected:
            raise ValueError('assigned ranking and predictive guard treatment required')
    treatment=dict(arm=arm,selected_plans=len(plans),actual_treatment_verified=bool(plans),
        external_neural_correction_applied=False,neural_reference_unused_by_control=arm in ('pose_command','command_history'))
    save_or_read(root,'actual_controller_treatment_v1.json',treatment)
    physical=save_or_read(root,'continuous_native_arrival_evaluation.json',arrivals(root))
    summary=save_or_read(root,'live_navigation_summary_v1.json',summarize(root))
    if arm!='reactive':
        windows=save_or_read(root,'saved_executed_motion_forecast_evaluation_v1.json',xy(root))
        save_or_read(root,'saved_short_pulse_yaw_evaluation_v1.json',yaw_metrics(root,windows))
        save_or_read(root,'saved_short_pulse_same_window_xy_v1.json',same_window_xy(root,windows))
    result=dict(assignment=number,arm=arm,layout_index=index,round_trip=physical['round_trip_arrival_checks_passed'],
        contacts=physical['disallowed_contact_samples'],arrivals=physical['arrivals'],
        terminal=physical['mission_terminal'],failure=summary['failure'],
        plans=len(plans),plans_on_time=summary['plans_on_time'],
        simulation_s=None if summary['result'] is None else summary['result']['simulation_s'])
    save_or_read(root,'short_pulse_navigation_evaluation_v1.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--assignment',type=int,choices=range(1,15),required=True)
    evaluate(parser.parse_args().assignment)
