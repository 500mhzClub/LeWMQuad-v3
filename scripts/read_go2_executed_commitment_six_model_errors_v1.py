"""Six fixed predictors on common actually executed half-second commitments."""
from collections import deque
import hashlib
import json
import math
import time
import cv2
import numpy as np
import torch
from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.geometry_progress_predictive_selection_development import select
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.physical_execution_development import rotation_xyzw
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.matched_family_model_admission_development import load_assigned, FITS, ROSTER
from scripts.family_transition_model_admission_development import admit
from scripts.read_go2_active_view_six_model_transitions_v1 import FIT_SHA
from scripts.read_go2_exact_terminal_target_v1 import (
    INPUT, READOUT, INPUT_SHA, READOUT_SHA, CASE, BASE, create_output,
    validate_root, verify_artifacts, digest, write_json, read_json, discover_sources, verify, hardware)

OUTPUT=BASE/'go2_executed_commitment_six_model_errors_v1_attempt_001'
PROTOCOL='docs/go2_executed_commitment_six_model_errors_v1_2026-09-08.md'


def relative_motion(first,last):
    a,b=np.asarray(first,float),np.asarray(last,float)
    if a.shape!=(7,) or b.shape!=(7,) or not np.isfinite([a,b]).all():
        raise ValueError('finite native poses required for evaluator')
    R=rotation_xyzw(a[3:]);delta=R.T@(b[:3]-a[:3]);relative=R.T@rotation_xyzw(b[3:])
    if np.hypot(relative[1,0],relative[0,0])<=1e-8:raise ValueError('undefined relative body yaw')
    return [float(delta[0]),float(delta[1]),float(np.arctan2(relative[1,0],relative[0,0]))]


def contexts():
    reader=IntentReturnRGBDReplay(INPUT/CASE);rows=read_json(INPUT/CASE,'context_decisions.json')
    assert len(reader.frames)==240 and len(rows)==240
    history=deque(maxlen=4);result=[]
    for frame in range(240):
        p,d,f,now=reader.packet(frame);history.append(p)
        selection=rows[frame]['decision']['new_selection']
        if selection is None:continue
        tensors=causal_history_tensors(list(history),now)
        ids={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()) for k,v in tensors.items()}
        result.append(dict(tick=frame,now_ns=now,selection=selection,tensors=tensors,input_sha256=ids,
            observed_goal_distance_m=rows[frame]['decision']['observed_goal_distance_m']))
    assert len(result)==46
    return result


def predict_all(launch, inputs):
    products={};reports={}
    for name in ROSTER:
        model,condition,variant=load_assigned(launch,name);before=state_digest(model.state_dict());rows=[]
        for context in inputs:
            old=context['selection']
            prediction=select(model,context['tensors'],head='direct_outcomes' if condition=='direct' else 'rollout_outcomes',
                input_variant=variant,goal_body_xy_m=old['goal_body_xy_m'],contact_penalty_m=1.2)
            if name=='seed_2026091001_full_direct':assert prediction['prediction']==old['prediction']
            rows.append(dict(tick=context['tick'],input_sha256=context['input_sha256'],
                prediction=prediction['prediction'],selection_wall_ms=prediction['selection_wall_ms'],
                head=prediction['head'],input_variant=variant))
        assert state_digest(model.state_dict())==before
        filename=name+'_predictions.json';write_json(OUTPUT/filename,rows);products[filename]=digest(OUTPUT/filename)
        reports[name]=dict(condition=condition,variant=variant,model_state_sha256=before,
            state_unchanged=True,contexts=len(rows),prediction_file=filename,
            original_all_six_forecasts_exact=name=='seed_2026091001_full_direct')
        print('EXECUTED_COMMITMENT_PREDICTED',name,len(rows),flush=True)
        del model
    return products,reports


def evaluate(inputs,reports):
    tape=read_json(INPUT/CASE,'command_tape.json');cameras=read_json(INPUT/CASE,'camera_audit.json')
    with np.load(INPUT/CASE/'physics_trace.npz',allow_pickle=False) as archive:
        poses=archive['base_pose_world']
    predictions={name:read_json(OUTPUT,r['prediction_file']) for name,r in reports.items()}
    rows=[];excluded=[];previous=None
    for number,context in enumerate(inputs):
        tick=context['tick'];s=context['selection'];action=s['action'];command=candidate_commands(action)[:5]
        interval=tape[tick:tick+5]
        complete=len(interval)==5 and all(t['completed'] and t['requested_command']==list(c) for t,c in zip(interval,command,strict=True))
        group='startup' if previous is None else 'repeat' if previous==action else 'switch';previous=action
        if not complete:
            excluded.append(dict(tick=tick,action=action,reason='selected five-command commitment interrupted'));continue
        start,end=cameras[tick]['physical_sample_index'],cameras[tick+5]['physical_sample_index']
        assert end-start==250 and end<len(poses)
        actual=relative_motion(poses[start],poses[end]);errors={}
        for name,values in predictions.items():
            assert values[number]['tick']==tick and values[number]['input_sha256']==context['input_sha256']
            p=np.asarray(values[number]['prediction'],float)[ACTIONS.index(action),0]
            if np.hypot(p[2],p[3])<=1e-8:raise ValueError('undefined forecast yaw')
            yaw=math.atan2(p[2],p[3]);dyaw=math.atan2(math.sin(yaw-actual[2]),math.cos(yaw-actual[2]))
            errors[name]=dict(predicted_body_xy_m=p[:2].tolist(),predicted_yaw_rad=yaw,
                translation_error_m=float(np.linalg.norm(p[:2]-actual[:2])),yaw_error_rad=abs(dyaw),
                signed_xy_error_m=(p[:2]-actual[:2]).tolist(),signed_yaw_error_rad=dyaw,
                uncalibrated_contact_score=float(np.exp(-np.logaddexp(0.,-p[4]))))
        rows.append(dict(tick=tick,mode=s['mode'],action=action,transition=group,
            near_goal=context['observed_goal_distance_m']<=.35,
            observed_goal_distance_m=context['observed_goal_distance_m'],
            native_start_sample=start,native_end_sample=end,actual_body_xy_yaw=actual,
            errors=errors,only_executed_action_labeled=True,contact_calibration_evaluated=False))
    assert len(rows)==45 and [r['tick'] for r in excluded]==[228]
    summary=[]
    for name in ROSTER:
        for scope in ('all','waypoint','near_goal','switch','repeat','startup'):
            chosen=[r for r in rows if scope=='all' or scope=='waypoint' and r['mode']=='WAYPOINT'
                or scope=='near_goal' and r['near_goal'] or r['transition']==scope]
            summary.append(dict(model=name,scope=scope,commitments=len(chosen),
                mean_translation_error_m=float(np.mean([r['errors'][name]['translation_error_m'] for r in chosen])) if chosen else None,
                mean_yaw_error_rad=float(np.mean([r['errors'][name]['yaw_error_rad'] for r in chosen])) if chosen else None,
                maximum_translation_error_m=max((r['errors'][name]['translation_error_m'] for r in chosen),default=None),
                maximum_yaw_error_rad=max((r['errors'][name]['yaw_error_rad'] for r in chosen),default=None)))
    return dict(commitments=rows,excluded=excluded,summary=summary,
        independent_trajectories=1,optimization_seeds=1,policy_owner='seed_2026091001_full_direct',
        matched_control_comparison=False,unexecuted_actions_labeled=False,model_selected=False)


def main():
    if not __debug__:raise ValueError('enabled assertions required')
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive executed-commitment readout')
    resources=hardware()
    if resources['memory_available_bytes']<8*1024**3 or resources['artifact_free_bytes']<40*1024**3+256*1024**2:
        raise ValueError('readout resource allowance unavailable')
    verify_artifacts(INPUT,{'result.json':INPUT_SHA});prior=read_json(INPUT,'result.json')
    assert prior['status']=='OVERLAP_RETENTION_GOAL_PROBE_COMPLETE'
    ids={'result.json':INPUT_SHA}|prior['artifact_sha256'];verify_artifacts(INPUT,ids)
    verify_artifacts(READOUT,{'result.json':READOUT_SHA});readout=read_json(READOUT,'result.json')
    assert readout['probe_result_sha256']==INPUT_SHA
    readout_ids={'result.json':READOUT_SHA,'launch.json':readout['launch_sha256']}
    verify_artifacts(READOUT,readout_ids);original=read_json(READOUT,'launch.json');verify(original)
    _,admission=admit(FIT_SHA)
    sources=discover_sources((PROTOCOL,'scripts/read_go2_executed_commitment_six_model_errors_v1.py',
        'lewm/tests/test_executed_commitment_labels_development.py',
        'docs/go2_adjacent_pair_correspondence_result_2026-09-08.md',
        'docs/go2_family_transition_fits_result_2026-09-08.md'),original['source_sha256'])
    launch=original|dict(source_sha256=sources,output_root=str(OUTPUT),protocol=PROTOCOL,
        probe_artifact_sha256=ids,readout_artifact_sha256=readout_ids,hardware=resources,
        all_six_admission=admission,native_execution=False,workers=1,threads=1,
        concurrency_reason='276 small predictions; sequential fixed-model readout, no native or training job',
        frozen_model_roster=list(ROSTER),native_labels_opened_after_all_predictions=True)
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('EXECUTED_COMMITMENT_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True);started=time.perf_counter()
    try:
        inputs=contexts();products,reports=predict_all(launch,inputs)
        write_json(OUTPUT/'prediction_phase_complete.json',dict(reports=reports,artifact_sha256=products,native_labels_parsed=False))
        products['prediction_phase_complete.json']=digest(OUTPUT/'prediction_phase_complete.json')
        verify_artifacts(OUTPUT,products)
        evaluation=evaluate(inputs,reports);write_json(OUTPUT/'evaluation.json',evaluation)
        products.update({n:digest(OUTPUT/n) for n in ('launch.json','evaluation.json')})
        verify(launch);verify_artifacts(INPUT,ids);verify_artifacts(READOUT,readout_ids)
        verify_artifacts(FITS,launch['fit_artifact_sha256']);verify_artifacts(OUTPUT,products)
        write_json(OUTPUT/'result.json',dict(status='EXECUTED_COMMITMENT_SIX_MODEL_ERRORS_COMPLETE',
            source_sha256=sources,artifact_sha256=products,reports=reports,summary=evaluation['summary'],
            completed_commitments=45,interrupted_commitments=evaluation['excluded'],
            hardware_after=hardware(),wall_s=time.perf_counter()-started,
            native_execution=False,model_training=False,model_selected=False,matched_control_comparison=False,
            original_outcome_changed=False,navigation_qualified=False,goal_achieved=False))
        print('EXECUTED_COMMITMENT_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_EXECUTED_COMMITMENT_READOUT_FAILURE',reason=repr(error)))
        raise


if __name__=='__main__':main()
