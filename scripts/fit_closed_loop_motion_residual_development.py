"""CPU development study: causal visual-motion features correct frozen JEPA XY forecasts."""
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from scripts.in_memory_public_replay_development import PublicReplay
from scripts.run_go2_paced_native_prefix_development import BASE,load_assigned
from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.delayed_action_planning_development import delayed_candidate_inputs
from lewm.observation_horizon_input_ablation_development import transform_inputs
from lewm.geometry_progress_pilot_development import ACTIONS,candidate_commands

TRAIN=(
    'go2_frontier_visit_native_layout00_v1_attempt_001',
    'go2_translation_reserve_native_layout00_v1_attempt_001',
    'go2_clearance_lookahead_native_layout00_v1_attempt_001',
    'go2_pruned_clearance_lookahead_native_layout00_v1_attempt_001')
VALIDATION='go2_standoff_frontier_native_layout00_v1_attempt_001'
OUTPUT=BASE/'go2_closed_loop_motion_residual_study_v1_attempt_001'


def features(prediction,past,commands):
    """Horizon h sees past poses, its base forecast, and commands only through h."""
    values=[]
    for h in range(8):
        known=np.zeros((8,3));known[:h+1]=commands[:h+1]
        yaw=np.cumsum(known[:,2]*.1)-known[:,2]*.05
        nominal=np.array([np.sum(known[:,0]*np.cos(yaw))*.1,np.sum(known[:,0]*np.sin(yaw))*.1])
        values.append(np.r_[past,prediction[h,:2],prediction[h,2:4]-[0.,1.],known.ravel(),nominal])
    return np.asarray(values)


def pose_features(poses,frame):
    p=np.asarray(poses[frame]['position_initial_body_m'])
    R=np.asarray(poses[frame]['rotation_initial_body_from_current_body'])
    past=[]
    for f in range(frame-3,frame):
        delta=R.T@(np.asarray(poses[f]['position_initial_body_m'])-p)
        rotation=R.T@np.asarray(poses[f]['rotation_initial_body_from_current_body'])
        yaw=np.arctan2(rotation[1,0],rotation[0,0])
        past.extend([*delta[:2],np.sin(yaw),np.cos(yaw)-1.])
    return np.asarray(past),p,R


@torch.inference_mode()
def collect(root_name,model,variant,training):
    root=BASE/root_name;reader=PublicReplay(root/'native')
    poses={r['frame']:r['registered_pose'] for r in json.loads((root/'poses.json').read_text())}
    requests={r['now_ns']:r['requested_command'] for r in json.loads((root/'requests.json').read_text())}
    plans=[r for r in json.loads((root/'planning.json').read_text()) if 'selection' in r]
    rows=[];xs=[];predictions=[];targets=[];masks=[]
    for plan in plans:
        frame=plan['frame'];now=plan['measured_ns']
        if any(f not in poses for f in range(frame-3,frame+9)):continue
        command=candidate_commands(plan['action'])[0]
        commands=np.asarray(plan['committed_prefix']+[command]*4+[[0.,0.,0.]])
        moving=bool(np.any(commands[:7]));transition=not np.array_equal(commands[2],commands[3])
        # Fixed thinning prevents long stationary tails dominating the fit.
        if training and not moving and frame%40:continue
        matches=[all(ns in requests and np.allclose(requests[ns],c,atol=1e-8,rtol=0)
            for ns in range(now+i*100_000_000,now+(i+1)*100_000_000,20_000_000)) for i,c in enumerate(commands)]
        valid=np.logical_and.accumulate(matches)
        if not valid.any():continue
        history=[reader.policy_packet(f) for f in range(frame-3,frame+1)]
        inputs=transform_inputs(delayed_candidate_inputs(causal_history_tensors(history,now),
            plan['committed_prefix'],delay_ticks=3,commit_ticks=4),input_variant=variant)
        output=model(**inputs)
        if not output['prediction_valid'].all():raise ValueError('complete known-command forecast required')
        prediction=output['rollout_outcomes'][ACTIONS.index(plan['action'])].cpu().numpy()
        past,p,R=pose_features(poses,frame)
        target=np.array([(R.T@(np.asarray(poses[f]['position_initial_body_m'])-p))[:2]
            for f in range(frame+1,frame+9)])
        xs.append(features(prediction,past,commands));predictions.append(prediction[:,:2]);targets.append(target);masks.append(valid)
        translation=bool(np.any(commands[:7,0]))
        rows.append(dict(frame=frame,moving=moving,transition=transition,translation=translation,
            turn_only=moving and not translation,action=plan['action']))
    result=dict(features=np.asarray(xs),prediction=np.asarray(predictions),target=np.asarray(targets),valid=np.asarray(masks))
    np.savez_compressed(OUTPUT/(root_name+'.npz'),**result)
    with (OUTPUT/(root_name+'.json')).open('x') as f:json.dump(rows,f)
    print('COLLECTED',root_name,len(rows),'valid700',int(result['valid'][:,6].sum()),flush=True)
    return result,rows


def metrics(errors):
    distance=np.linalg.norm(errors,axis=-1)
    if not len(distance):return dict(count=0)
    return dict(count=len(distance),rmse_m=float(np.sqrt(np.mean(distance**2))),
        median_m=float(np.median(distance)),p90_m=float(np.percentile(distance,90)),
        p95_m=float(np.percentile(distance,95)),maximum_m=float(distance.max()))


def main():
    if OUTPUT.exists():raise ValueError('preserve previous study')
    OUTPUT.mkdir()
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    began=time.time()
    admission=json.loads((BASE/'go2_all_phase_adapter_maze02_matched_native_v1_attempt_001/launch.json').read_text())['input_admission']['correction_admission']
    model,condition,variant=load_assigned(admission,'seed_2026091001_full_jepa')
    with (OUTPUT/'launch.json').open('x') as f:json.dump(dict(training_roots=TRAIN,validation_root=VALIDATION,
        labels='subsequent_registered_visual_pose_deltas',native_state_read=False,
        base_model='seed_2026091001_full_jepa',ridge_penalty=1.,stationary_training_stride_frames=40,
        latest_run_excluded_from_fit=True,validation_is_development_not_sealed=True),f,indent=2)
    datasets=[collect(root,model,variant,True)[0] for root in TRAIN]
    train={k:np.concatenate([d[k] for d in datasets]) for k in datasets[0]}
    coefficients=[];means=[];scales=[];biases=[]
    for h in range(8):
        valid=train['valid'][:,h];x=train['features'][valid,h];y=(train['target']-train['prediction'])[valid,h]
        mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1.;bias=y.mean(0)
        z=(x-mean)/scale;coefficient=np.linalg.solve(z.T@z+np.eye(z.shape[1]),z.T@(y-bias))
        means.append(mean);scales.append(scale);biases.append(bias);coefficients.append(coefficient)
    fit=dict(mean=np.asarray(means),scale=np.asarray(scales),bias=np.asarray(biases),coefficient=np.asarray(coefficients))
    np.savez_compressed(OUTPUT/'residual_fit.npz',**fit)
    # Coefficients are frozen before reading the validation forecast/target windows.
    validation,rows=collect(VALIDATION,model,variant,False)
    corrected=validation['prediction'].copy()
    for h in range(8):
        corrected[:,h]+=((validation['features'][:,h]-fit['mean'][h])/fit['scale'][h])@fit['coefficient'][h]+fit['bias'][h]
    results={}
    for group in ('all','moving','transition','translation','turn_only'):
        group_mask=np.array([True if group=='all' else row[group] for row in rows])
        results[group]={}
        for h in (2,6,7):
            mask=group_mask&validation['valid'][:,h]
            results[group][str((h+1)*100)]=dict(base=metrics((validation['prediction']-validation['target'])[mask,h]),
                corrected=metrics((corrected-validation['target'])[mask,h]))
    report=dict(status='COMPLETE',results=results,training_windows=len(train['features']),validation_windows=len(rows),
        elapsed_s=time.time()-began,native_state_read=False,neural_weights_changed=False,
        inference_uses_causal_pose_history_and_known_commands=True,prospective_navigation_tested=False)
    with (OUTPUT/'result.json').open('x') as f:json.dump(report,f,indent=2)
    print('MOTION_RESIDUAL_RESULT',json.dumps(report),flush=True)


if __name__=='__main__':main()
