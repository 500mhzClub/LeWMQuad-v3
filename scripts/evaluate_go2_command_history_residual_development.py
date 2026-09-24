"""Paired predictions from reference-only, learned corrections and old models."""
from functools import lru_cache
import hashlib
import json
import math
import time
import cv2
import numpy as np
import torch
from lewm.eligible_floor_registration_development import bind
from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.delayed_action_planning_development import delayed_candidate_inputs
from lewm.terminal_translation_pulse_development import command_sequences
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.physical_execution_development import rotation_xyzw
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts import train_go2_command_history_residual_development as train
from scripts import command_history_residual_snapshot_development as snapshot
from scripts import nominal_motion_residual_snapshot_development as old_snapshot
from scripts import evaluate_go2_longer_residual_fit_development as previous
from scripts import read_go2_action_forecast_bias_development as bias

OUTPUT=train.OUTPUT/'prediction_evaluation'
VARIANTS=('reference_only','jepa_reference_residual','supervised_rollout_reference_residual','jepa_1200','supervised_rollout_1200')
metrics=bind(previous.metrics,VARIANTS=VARIANTS)


@torch.inference_mode()
def main():
    if OUTPUT.exists():raise ValueError('preserve every evaluation')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    plan=json.loads(train.PLAN.read_text());models={};identities={}
    for condition in train.CONDITIONS:
        for new in (False,True):
            directory=(train.OUTPUT/condition if new else train.data.BASE/'go2_short_pulse_residual_matched_fits_v1_attempt_001')
            record=json.loads((directory/'fit.json').read_text()) if new else plan['old_models'][condition]
            name=condition+('_reference_residual' if new else '_1200');loader=snapshot if new else old_snapshot
            assert record['configuration']['updates']==1200
            models[name]=loader.load_snapshot(directory,record['filename'],sha256=record['sha256'],
                expected_binding=record['binding'],expected_config=record['configuration']).model
            identities[name]=dict(directory=str(directory),sha256=record['sha256'],model_sha256=record['model_sha256'])
    identities['reference_only']=dict(path=plan['reference_path'],sha256=plan['reference_sha256'])
    OUTPUT.mkdir();began=time.monotonic();labels=json.loads((bias.OUTPUT/'result.json').read_text())
    assert labels['sources']==[str(train.BASE/r) for r in plan['evaluation_roots']]
    all_rows=[];runs=[];conflicts=[];max_old=0.;max_reference=0.
    for number,root in enumerate(bias.SOURCES,1):
        read=lambda name:json.loads((root/name).read_text())
        windows={r['frame']:r for r in labels['rows'] if r['run']==number}
        plans={r['frame']:r for r in read('planning.json') if 'selection' in r}
        old_arm=read('launch.json')['training_condition'];reader=NoisyPublicReplay(root/'native')
        packet=lru_cache(maxsize=8)(reader.policy_packet)
        camera={r['frame']:r for r in read('native/in_memory_camera_observations.json')['frames']}
        with np.load(root/'native/physics_trace.npz',allow_pickle=False) as data:physics=data['base_pose_world'].copy()
        extra=(504,508,512,516,520,524,528) if number==6 else ();rows=[]
        for ordinal,frame in enumerate(sorted(set(windows)|set(extra)),1):
            p=plans[frame];c=p['motion_correction']
            history=causal_history_tensors([packet(i) for i in range(frame-3,frame+1)],p['measured_ns'])
            inputs=delayed_candidate_inputs(history,p['committed_prefix'],delay_ticks=3,commit_ticks=4)
            if c['terminal_translation_pulse']:
                inputs['known_action_blocks']=torch.as_tensor(command_sequences(p['committed_prefix'],pulse=True)[:,:,None],
                    dtype=torch.float32)/torch.tensor([.3,1.,.5])
            predictions={name:model(**inputs)['rollout_outcomes'].cpu().numpy() for name,model in models.items()}
            ref=models['jepa_reference_residual'].reference_motion(inputs['observation_history'],inputs['known_action_blocks'],inputs['known_action_valid']).numpy()
            predictions['reference_only']=np.concatenate((ref[:,:,:2],np.sin(ref[:,:,2:3]),np.cos(ref[:,:,2:3]),np.zeros((6,8,1))),axis=-1)
            recorded=np.asarray(c['command_history_forecast_xy_yaw'])
            difference=float(np.max(np.abs(ref-recorded)));max_reference=max(max_reference,difference)
            np.testing.assert_allclose(ref,recorded,rtol=0,atol=2e-6)
            old=predictions[f'{old_arm}_1200'][:,:,:4];recorded_old=np.asarray(c['upstream_prediction_for_yaw_ablation'])[:,:,:4]
            np.testing.assert_allclose(old,recorded_old,rtol=0,atol=1e-6)
            max_old=max(max_old,float(np.max(np.abs(old-recorded_old))))
            assert all(v.shape==(6,8,5) and np.isfinite(v).all() for v in predictions.values())
            if frame in extra:conflicts.append(dict(frame=frame,predictions={k:v[:,:,:4].tolist() for k,v in predictions.items()}))
            if frame in windows:
                w=windows[frame];index=ACTIONS.index(w['action']);errors={};yaw_errors={}
                endpoints=[rotation_xyzw(physics[camera[f]['physical_sample_index'],3:]) for f in (frame,frame+7)]
                actual_yaw=math.atan2(endpoints[1][1,0],endpoints[1][0,0])-math.atan2(endpoints[0][1,0],endpoints[0][0,0])
                for name,prediction in predictions.items():
                    parts=bias.components(prediction[index,:,:2])
                    errors[name]={part:(1000*(np.asarray(parts[part])-w['actual_xy_m'][part])).tolist() for part in bias.PARTS}
                    delta=math.atan2(prediction[index,6,2],prediction[index,6,3])-actual_yaw
                    yaw_errors[name]=math.atan2(math.sin(delta),math.cos(delta))
                rows.append(dict(run=number,frame=frame,action=w['action'],prefix_group=w['prefix_group'],
                    recorded_arm=old_arm,errors_mm=errors,yaw_error_rad=yaw_errors))
            if ordinal%300==0:print('REFERENCE_EVALUATION',number,ordinal,flush=True)
        runs.append(dict(run=number,root=str(root),recorded_arm=old_arm,total=metrics(rows),
            by_action={a:metrics([r for r in rows if r['action']==a]) for a in ACTIONS}))
        all_rows.extend(rows);train.write(OUTPUT/f'run_{number:02d}.json',dict(summary=runs[-1],rows=rows))
        print('REFERENCE_EVALUATION_RUN_COMPLETE',number,len(rows),flush=True)
    result=dict(schema='command_history_residual_prediction_evaluation.v1',model_identities=identities,
        source_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest(),runs=runs,total=metrics(all_rows),
        by_action={a:metrics([r for r in all_rows if r['action']==a]) for a in ACTIONS},
        maximum_original_forecast_difference=max_old,maximum_command_reference_difference=max_reference,
        command_reference_difference_reason='float32 input tensor versus original double command history',
        all_six_preselected_runs_included=True,matched_input_windows_for_all_five_models=True,
        matched_requested_command_sequence_ms=700,original_training_data_only=True,navigation_reexecuted=False,
        RGB_body_command_only_model_inputs=True,depth_not_loaded=True,native_state_evaluator_only=True,
        overlapping_windows_not_independent=True,exposed_development_evaluation=True,
        independent_generalization_established=False,unexecuted_conflict_forecasts=conflicts,wall_s=time.monotonic()-began)
    train.write(OUTPUT/'result.json',result);print(json.dumps(result['total'],indent=2),flush=True)


if __name__=='__main__':main()
