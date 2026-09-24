"""Signed motion errors for every matched window in the six recent missions."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.physical_execution_development import rotation_xyzw
from lewm.commanded_planar_motion_development import forecast as nominal
from scripts import run_go2_view_replan_repeatability_development as repeated
from scripts import run_go2_earlier_visual_recovery_development as earlier

OUTPUT=earlier.BASE/'go2_action_forecast_bias_readout_v1_attempt_001'
SOURCES=[(repeated.BASE/repeated.root_name(n)) for n in range(1,5)]+[
    earlier.BASE/earlier.root_name(n) for n in (1,2)]
MODELS=('neural','pose_command','command_history','nominal')
PARTS=('prefix_300ms','whole_700ms','action_increment_400ms')


def metrics(rows):
    if not rows:return dict(windows=0)
    result=dict(windows=len(rows),parts={})
    for part in PARTS:
        actual=np.array([r['actual_xy_m'][part] for r in rows])
        values={}
        for model in MODELS:
            predicted=np.array([r['forecast_xy_m'][model][part] for r in rows])
            error=predicted-actual;norm=np.linalg.norm(error,axis=1)
            values[model]=dict(rmse_mm=float(1000*np.sqrt(np.mean(norm**2))),
                mean_signed_xy_error_mm=(1000*error.mean(0)).tolist(),
                p95_error_mm=float(1000*np.percentile(norm,95)),
                median_predicted_displacement_mm=float(1000*np.median(np.linalg.norm(predicted,axis=1))))
        result['parts'][part]=dict(median_actual_displacement_mm=float(
            1000*np.median(np.linalg.norm(actual,axis=1))),models=values)
    return result


def components(xy):
    return dict(prefix_300ms=xy[2].tolist(),whole_700ms=xy[6].tolist(),
        action_increment_400ms=(xy[6]-xy[2]).tolist())


def main():
    if OUTPUT.exists():raise ValueError('preserve complete or partial diagnostic')
    OUTPUT.mkdir();rows=[];runs=[]
    for number,root in enumerate(SOURCES,1):
        read=lambda name:json.loads((root/name).read_text())
        launch=read('launch.json');model=launch['training_condition']
        plans={p['frame']:p for p in read('planning.json') if 'selection' in p}
        windows=read('saved_executed_motion_forecast_evaluation_v1.json')
        assert windows['matched_requested_sequence_through_ns']==700_000_000
        frames={r['frame']:r for r in read('native/in_memory_camera_observations.json')['frames']}
        with np.load(root/'native/physics_trace.npz',allow_pickle=False) as data:physics=data['base_pose_world'].copy()
        population=[]
        for window in windows['rows']:
            frame=window['frame'];plan=plans[frame];index=ACTIONS.index(window['action'])
            origin=physics[frames[frame]['physical_sample_index']]
            future=physics[[frames[frame+h]['physical_sample_index'] for h in range(1,8)],:3]
            actual=((future-origin[:3])@rotation_xyzw(origin[3:]))[:,:2]
            np.testing.assert_allclose(actual[-1],window['actual_endpoint_xy_m'],rtol=0,atol=1e-12)
            correction=plan['motion_correction'];prefix=np.asarray(plan['committed_prefix'])
            forecasts=dict(neural=correction['raw_forecast_xy_m'],
                pose_command=correction['pose_command_forecast_xy_m'],
                command_history=correction['command_history_forecast_xy_yaw'],
                nominal=nominal(plan['committed_prefix'],pulse=bool(correction['terminal_translation_pulse'])))
            prefix_group=('translation' if np.any(np.abs(prefix[:,:2])>1e-8) else
                'mixed_turn' if np.any(prefix[:,2]>1e-8) and np.any(prefix[:,2]<-1e-8) else
                'left_turn' if np.any(prefix[:,2]>1e-8) else
                'right_turn' if np.any(prefix[:,2]<-1e-8) else 'zero')
            row=dict(run=number,arm=model,frame=frame,action=window['action'],prefix_group=prefix_group,
                visual_recovery=plan['route_status']=='LOW_VISUAL_SUPPORT_REQUIRES_MEASURED_VIEW',
                actual_xy_m=components(actual),forecast_xy_m={k:components(np.asarray(v)[index,:,:2])
                    for k,v in forecasts.items()})
            population.append(row)
        total=metrics(population)
        previous=read('saved_short_pulse_same_window_xy_v1.json')
        for model_name in MODELS:
            np.testing.assert_allclose(total['parts']['whole_700ms']['models'][model_name]['rmse_mm'],
                previous['rmse_mm'][model_name],rtol=0,atol=1e-9)
        runs.append(dict(run=number,root=str(root),arm=model,round_trip=read('short_pulse_navigation_evaluation_v1.json')['round_trip'],
            total=total,by_action={a:metrics([r for r in population if r['action']==a]) for a in ACTIONS},
            by_action_and_prefix={a+'/'+g:metrics([r for r in population if r['action']==a and r['prefix_group']==g])
                for a,g in sorted({(r['action'],r['prefix_group']) for r in population})}))
        rows.extend(population);print('ACTION_FORECAST_RUN',number,len(population),flush=True)
    pooled={arm:{action:metrics([r for r in rows if r['arm']==arm and r['action']==action])
        for action in ACTIONS} for arm in ('jepa','supervised_rollout')}
    result=dict(schema='action_forecast_bias_readout.v1',sources=list(map(str,SOURCES)),runs=runs,rows=rows,
        pooled_by_arm_and_action=pooled,matched_existing_whole_window_metrics=True,
        native_state_evaluator_only=True,all_recent_six_missions_included=True,
        incremental_displacement_axes='body axes at original planning observation, not rotated at action start',
        requested_not_actuator_applied_sequence_matching=True,overlapping_windows_not_independent=True,
        selected_executed_actions_only=True,unexecuted_recovery_turn_accuracy_unproven=True,
        new_training_or_online_correction=False)
    repeated.save(OUTPUT/'result.json',result)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
    labels=('left_turn','right_turn');xs=np.arange(2);width=.19
    for ax,part,title in zip(axes,('prefix_300ms','action_increment_400ms'),('Committed prefix: 300 ms','Following action increment: 400 ms')):
        for i,model_name in enumerate(MODELS):
            values=[pooled['jepa'][a]['parts'][part]['models'][model_name]['rmse_mm'] for a in labels]
            ax.bar(xs+(i-1.5)*width,values,width,label=model_name.replace('_',' '))
        ax.set_xticks(xs,('Left turn','Right turn'));ax.set_title(title);ax.set_ylabel('Planar prediction RMSE (mm)')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[1].legend(fontsize=8)
    fig.suptitle('JEPA-run turn forecasts: errors before and during the new action')
    fig.text(.5,.01,'Four JEPA missions, successes and failures; selected executed windows overlap.\n'
        'Pooled diagnostics do not establish alternative navigation outcomes.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.09,1,.95));fig.savefig(OUTPUT/'turn_forecast_errors.png',dpi=170)
    fig.savefig(OUTPUT/'turn_forecast_errors.svg');plt.close(fig)
    for arm in pooled:
        for action in labels:print(arm,action,json.dumps(pooled[arm][action]),flush=True)


if __name__=='__main__':main()
