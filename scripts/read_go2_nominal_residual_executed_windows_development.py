"""Exposed closed-loop replay: new predictions on fixed, actually executed tapes.

Uses retained RGB/body packets only; neither depth nor native state is read.
Truth comes from completed evaluator records, never from model input packets.
"""
from collections import defaultdict
import hashlib
import json
import time
import cv2
import numpy as np
import torch
from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.nominal_motion_residual_learning_development import nominal_motion
from lewm.terminal_translation_pulse_development import command_sequences
from scripts.public_policy_replay_development import PublicPolicyReplay
from scripts.evaluate_go2_nominal_motion_residual_development import roster,load
from scripts.read_go2_training_execution_coverage_development import BASE

ROOT_NAMES=tuple(f'go2_neural_rgb_transfer_seed_2026091001_full_jepa_noise_2mm_native_layout{i:02d}_4800_v1_attempt_001'
                 for i in (0,1))
OUTPUT=BASE/'go2_nominal_residual_executed_windows_v1_attempt_001'


def read(path):return json.loads(path.read_text())


def metrics(rows):
    report={}
    for name in rows[0]['errors'] if rows else ():
        values=[r['errors'][name] for r in rows]
        report[name]=dict(windows=len(values),xy_rmse_mm=1000*float(np.sqrt(np.mean([v['xy_m']**2 for v in values]))),
            yaw_rmse_deg=float(np.degrees(np.sqrt(np.mean([v['yaw_rad']**2 for v in values])))))
    return report


@torch.inference_mode()
def main():
    if OUTPUT.exists():raise ValueError('preserve executed-window diagnostic')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    models={};bindings={}
    for name,directory,record in roster():
        trainer=load(directory,record);models[name]=(trainer.model,trainer.condition)
        bindings[name]=dict(root=directory.name,filename=record['filename'],sha256=record['sha256'])
    OUTPUT.mkdir();started=time.monotonic();rows=[];identities={}
    try:
        for root_name in ROOT_NAMES:
            root=BASE/root_name;reader=PublicPolicyReplay(root/'native')
            for name in ('planning.json','saved_executed_motion_forecast_evaluation_v1.json',
                         'saved_neural_command_yaw_evaluation_v1.json','native/policy_histories.npz'):
                identities[root_name+'/'+name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
            plans={p['frame']:p for p in read(root/'planning.json') if 'selection' in p}
            truth=read(root/'saved_executed_motion_forecast_evaluation_v1.json')
            yaw_truth={r['frame']:r for r in read(root/'saved_neural_command_yaw_evaluation_v1.json')['rows']}
            if truth['matched_requested_sequence_through_ns']!=700_000_000:
                raise ValueError('actual seven-tick command matching required')
            for window in truth['rows']:
                frame=window['frame'];p=plans[frame];correction=p['motion_correction']
                index=p['selection']['action_index'];history=[]
                for f in range(frame-3,frame+1):
                    path=root/'native'/f'rgb_{f:04d}.png'
                    identities[str(path.relative_to(BASE))]=hashlib.sha256(path.read_bytes()).hexdigest()
                    history.append(reader.policy_packet(f))
                obs=causal_history_tensors(history,p['measured_ns'])
                commands=command_sequences(p['committed_prefix'],pulse=correction['terminal_translation_pulse'])[index]
                blocks=torch.tensor(commands,dtype=torch.float32)[None,:,None]/torch.tensor([.3,1.,.5])
                valid=torch.ones(1,8,1,dtype=torch.bool)
                inp=dict(observation_history={k:v[None] for k,v in obs.items()},
                         known_action_blocks=blocks,known_action_valid=valid)
                predictions={}
                for name,(model,condition) in models.items():
                    out=model(**inp)['direct_outcomes' if condition=='direct' else 'rollout_outcomes'][0,6].numpy()
                    predictions[name]=[*out[:2],float(np.arctan2(out[2],out[3]))]
                nominal=nominal_motion(blocks,valid)[0,6].numpy()
                predictions['command_integrated']=nominal
                predictions['saved_corrected_neural']=[*correction['learned_corrected_forecast_xy_m'][index][6],
                    float(np.arctan2(correction['upstream_prediction_for_yaw_ablation'][index][6][2],
                                     correction['upstream_prediction_for_yaw_ablation'][index][6][3]))]
                predictions['saved_fitted_xy_command_yaw']=[*correction['pose_command_forecast_xy_m'][index][6],nominal[2]]
                actual=np.asarray(window['actual_endpoint_xy_m'])
                angle=yaw_truth[frame]['actual_endpoint_yaw_rad']
                errors={}
                for name,pred in predictions.items():
                    dyaw=float(pred[2])-angle
                    errors[name]=dict(xy_m=float(np.linalg.norm(np.asarray(pred[:2])-actual)),
                                      yaw_rad=float(np.arctan2(np.sin(dyaw),np.cos(dyaw))))
                rows.append(dict(root_name=root_name,frame=frame,group=window['group'],action=window['action'],errors=errors))
            print('EXECUTED_READOUT_COMPLETE',root_name,len(truth['rows']),flush=True)
        result=dict(status='COMPLETE',roots=list(ROOT_NAMES),matched_windows=len(rows),pooled=metrics(rows),
            by_root={name:metrics([r for r in rows if r['root_name']==name]) for name in ROOT_NAMES},
            by_action_group={group:metrics([r for r in rows if r['group']==group]) for group in sorted({r['group'] for r in rows})},
            model_bindings=bindings,input_sha256=identities,wall_s=time.monotonic()-started,
            scope=dict(exposed_development_replay=True,independent_new_mazes=0,
                overlapping_windows=True,matched_horizon_ms=700,unexecuted_candidates_evaluated=False,
                predicted_alternative_navigation_outcomes=False,depth_read=False,native_artifacts_read=False,
                native_truth_from_completed_evaluators_only=True,
                yaw_truth_definition='saved wrapped world-heading change, as in prior pilot yaw readout'))
        (OUTPUT/'rows.json').write_text(json.dumps(rows)+'\n')
        (OUTPUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ('status','matched_windows','pooled','by_action_group','wall_s')},indent=2))
    except Exception as error:
        (OUTPUT/'failure.json').write_text(json.dumps(dict(reason=repr(error))))
        raise


if __name__=='__main__':main()
