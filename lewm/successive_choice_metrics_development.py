"""Evaluation-only reductions; no physical state is supplied to the policy."""
import math

import numpy as np

from lewm.counterfactual_maze_development import horizon_labels
from lewm.physical_execution_development import rotation_xyzw
from lewm.online_temporal_choice_development import METHODS


def reduce_trial(raw,*,start_index,tape,selections,direction,branchable,stop_reason,sensor_fault):
    times=np.rint(raw['timestamp_s']*1e9).astype(np.int64)
    origin=raw['base_pose_world'][start_index]; rotation=rotation_xyzw(origin[3:])
    unit=np.asarray(direction,dtype=float)/.8
    if unit.shape!=(2,) or not np.isclose(np.linalg.norm(unit),1,atol=1e-8,rtol=0): raise ValueError('initial direction')
    control=[e for e in tape if e['stage']=='control' and e['post_sample_index']>e['pre_sample_index']]
    end=control[-1]['post_sample_index'] if control else start_index
    delta=rotation.T@(raw['base_pose_world'][end,:3]-origin[:3])
    actions=[s['selected_action_index'] for s in selections]
    decisions=[]
    for index,selection in enumerate(selections):
        executed=[e for e in control if e['decision_index']==index]
        if not executed:
            decisions.append({'decision_index':index,'label':None,'position_error_m':None,'yaw_error_rad':None,'contact_brier':None})
            continue
        start=executed[0]['pre_sample_index']; terminal=executed[-1]['post_sample_index']
        label=horizon_labels({k:v[:terminal+1] for k,v in raw.items()},start)[0]
        # Motion is strictly pre-contact. The contact bit remains observable
        # when an early emergency stop makes the endpoint unavailable.
        if label['contact_by_horizon']:
            label['motion_valid']=False; label['delta_xy_yaw_start_body']=None
        position_error=yaw_error=brier=None
        action=selection['selected_action_index']
        if selection['mean_motion_sin_cos'] is not None:
            prediction=np.asarray(selection['mean_motion_sin_cos'][action])
            if label['motion_valid']:
                target=np.asarray(label['delta_xy_yaw_start_body'])
                position_error=float(np.linalg.norm(prediction[:2]-target[:2]))
                difference=math.atan2(prediction[2],prediction[3])-target[2]
                yaw_error=abs(math.atan2(math.sin(difference),math.cos(difference)))
            if label['contact_valid']:
                brier=float((selection['mean_contact_probability'][action]-label['contact_by_horizon'])**2)
        decisions.append({'decision_index':index,'label':label,'position_error_m':position_error,
            'yaw_error_rad':yaw_error,'contact_brier':brier})
    release=[e for e in tape if e['stage'] in ('release','fault_release')]
    release_complete=bool(len(release)==5 and all(e['post_sample_index']-e['pre_sample_index']==50 for e in release)
        and len(times)-1==release[-1]['post_sample_index'])
    release_pass=False; max_speed=max_wz=None
    if release_complete:
        tail=raw['base_twist_world'][-150:]
        max_speed=float(np.linalg.norm(tail[:,:2],axis=1).max()); max_wz=float(np.abs(tail[:,5]).max())
        release_pass=bool(max_speed<=.1 and max_wz<=.25)
    completed=bool(branchable and stop_reason is None and sensor_fault is None and len(control)==40
        and times[end]-times[start_index]==4_000_000_000 and release_complete)
    return {'any_contact':bool(raw['physics_contact'].any()),'prefix_failure':not branchable,
        'sensor_failure':sensor_fault is not None,'complete_control_and_release':completed,
        'observed_control_duration_s':float((times[end]-times[start_index])/1e9),
        'observed_signed_control_displacement_m':float(delta[:2]@unit) if branchable else None,
        'observed_lateral_control_displacement_m':float(delta[:2]@[-unit[1],unit[0]]) if branchable else None,
        'four_second_signed_displacement_m':float(delta[:2]@unit) if completed else None,
        'release_complete':release_complete,'release_motion_pass':release_pass,
        'release_max_xy_speed_mps':max_speed,'release_max_abs_world_wz_radps':max_wz,
        'selected_actions':actions,'selected_stop_count':actions.count(0),'decisions':len(actions),
        'action_changes':sum(a!=b for a,b in zip(actions,actions[1:])),
        'moving_to_stop':sum(a!=0 and b==0 for a,b in zip(actions,actions[1:])),
        'moving_to_reverse':sum(a not in (0,4) and b==4 for a,b in zip(actions,actions[1:])),
        'executed_decision_errors':decisions}


def paired_reduction(rows):
    layouts=[]
    for layout in sorted({r['layout_id'] for r in rows}):
        methods={}
        for method in METHODS:
            selected=[r['metrics'] for r in rows if r['layout_id']==layout and r['method']==method]
            intents={r['intent_name'] for r in rows if r['layout_id']==layout and r['method']==method}
            if len(selected)!=3 or intents!={'forward','left','right'}: raise ValueError('incomplete paired panel')
            values={key:float(np.mean([m[key] for m in selected])) for key in (
                'any_contact','prefix_failure','sensor_failure','complete_control_and_release','release_motion_pass',
                'action_changes','moving_to_stop','moving_to_reverse','selected_stop_count','decisions')}
            for key in ('observed_signed_control_displacement_m','four_second_signed_displacement_m'):
                observed=[m[key] for m in selected if m[key] is not None]
                values[key]={'observed_trials':len(observed),'conditional_mean':float(np.mean(observed)) if observed else None}
            layouts.append({'layout_id':layout,'method':method,**values})
    comparisons={}
    pairs=[(m,'always_stop') for m in METHODS if m!='always_stop']+[
        ('supervised_direct','direct_direct'),('jepa_direct','supervised_direct'),
        ('jepa_rollout','supervised_rollout'),('jepa_rollout','jepa_direct'),('supervised_rollout','supervised_direct')]
    for a,b in pairs:
        comparison={}
        for key in ('any_contact','complete_control_and_release','release_motion_pass'):
            delta=np.array([next(r[key] for r in layouts if r['layout_id']==layout and r['method']==a)
                -next(r[key] for r in layouts if r['layout_id']==layout and r['method']==b)
                for layout in sorted({r['layout_id'] for r in rows})])
            rng=np.random.default_rng(2026091899)
            samples=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(1)
            comparison[key]={'mean_delta':float(delta.mean()),'per_layout':delta.tolist(),
                'descriptive_layout_bootstrap_95_percentile':np.quantile(samples,[.025,.975]).tolist()}
        # Progress comparisons only on explicitly matched completed intents;
        # report omissions, never substitute zero for missing post-stop motion.
        matched=[]; missing=0
        for layout in sorted({r['layout_id'] for r in rows}):
            values=[]
            for intent in ('forward','left','right'):
                left=next(r for r in rows if r['layout_id']==layout and r['method']==a and r['intent_name']==intent)
                right=next(r for r in rows if r['layout_id']==layout and r['method']==b and r['intent_name']==intent)
                x=left['metrics']['four_second_signed_displacement_m']; y=right['metrics']['four_second_signed_displacement_m']
                if x is None or y is None: missing+=1
                else: values.append(x-y)
            matched.append({'layout_id':layout,'matched_intents':len(values),'conditional_mean_delta_m':float(np.mean(values)) if values else None})
        comparison['completed_pair_progress']={'by_layout':matched,'omitted_intent_pairs':missing,
            'warning':'survivor-conditional progress, interpret with all-trial failures; not all-trial efficacy'}
        comparisons[f'{a}_minus_{b}']=comparison
    return {'layout_methods':layouts,'paired_comparisons':comparisons}
