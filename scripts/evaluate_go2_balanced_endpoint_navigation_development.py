"""Test the navigation endpoint interface against existing native branches."""
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from lewm.dense_endpoint_navigation_development import DenseEndpointMotion, endpoint_waypoint_scores
from lewm.dense_visual_motion_readout_development import pool_tokens
from scripts import train_go2_balanced_start_predictor_development as fit
from scripts import train_go2_dense_visual_motion_readout_development as readout
from scripts.evaluate_go2_dense_visual_motion_readout_development import metrics

RESULT=Path('docs/go2_balanced_endpoint_navigation_2026-09-18.json')
OUTPUT=fit.OUTPUT/'endpoint_navigation_evaluation'


@torch.inference_mode()
def main():
    assert not RESULT.exists() and not OUTPUT.exists()
    terminal=json.loads(fit.RESULT.read_text());assert terminal['status']=='COMPLETE'
    previous=json.loads(Path('docs/go2_dense_visual_motion_readout_branch_result_2026-09-17.json').read_text())
    assert previous['fit_sha256']==fit.digest(readout.OUTPUT/'result.json')
    OUTPUT.mkdir();torch.set_num_threads(4);started=time.monotonic()
    reference=fit.parent.reference;rows=reference.selected_rows();prepared=[reference.inputs(r) for r in rows]
    assert previous['trials']==[r['trial'] for r in rows]
    stats=json.loads((reference.CACHE/'proprio_v1/proprio_norm_stats.json').read_text())
    head=readout.load().cuda()
    models={arm:fit.load(arm).cuda() for arm in fit.ARMS}
    state=torch.load(fit.parent.OUTPUT/'action_latest.pt',map_location='cpu',weights_only=False)
    parent=reference.ProprioActionPredictor(use_proprio=False).cuda().eval().requires_grad_(False)
    parent.load_state_dict(state['model_state_dict'],strict=True);models['parent_action']=parent;del state
    adapters={arm:DenseEndpointMotion(model,head,stats['control_mean'],stats['control_std'],
                                     action_blind=arm.endswith('no_future_action')).cuda() for arm,model in models.items()}
    encoder=reference.encoders.VJepa21Arm();encoder.build(torch.device('cuda:0'),torch.float32)
    cache={}
    def encode(path):
        key=fit.digest(path)
        if key not in cache:
            value=encoder.tokens(encoder.preprocess(str(path))[None].cuda()).float()
            cache[key]=F.layer_norm(value,(1024,))[0]
        return cache[key]
    predictions={arm:[] for arm in adapters};contexts=[]
    for offset in range(0,len(rows),3):
        directories=[prepared[i][0] for i in range(offset,offset+3)]
        context=torch.stack([encode(directories[0]/f'rgb_{i:04d}.png') for i in (3,8,13)])
        for directory in directories[1:]:
            assert torch.equal(context,torch.stack([encode(directory/f'rgb_{i:04d}.png') for i in (3,8,13)]))
        contexts.extend([context[-1:]]*3)
        meta=json.loads((directories[0]/'policy_observations.json').read_text())
        with np.load(directories[0]/'policy_histories.npz',allow_pickle=False) as a:
            assert a['applied_command_valid'][13].all()
            past=a['applied_command_values'][13].astype(np.float32)
            measured=a['applied_command_measured_ns'][13].copy();available=a['applied_command_available_ns'][13].copy()
        future=np.zeros((3,5,3),np.float32)
        for j,i in enumerate(range(offset,offset+3)):future[j][:,[0,2]]=prepared[i][2].reshape(5,2)
        for arm,adapter in adapters.items():
            output=adapter(context_tokens=context,context_times_ns=[meta['frames'][i]['image_ns'] for i in (3,8,13)],
                           past_applied_commands=past,past_measured_ns=measured,past_available_ns=available,
                           future_applied_commands=future)
            assert output['target_offset_ns']==500_000_000 and not output['intermediate_motion_available']
            motion=output['endpoint_motion_body_xy_yaw']
            if arm.endswith('no_future_action'):assert torch.equal(motion,motion[:1].expand_as(motion))
            predictions[arm].append(motion.cpu().numpy())
    fit.save(OUTPUT/'causal_predictions_complete.json',dict(future_images_and_physical_targets_loaded=False,
                                                          contexts=len(rows),readout_refitted=False))
    truth=np.asarray([r['targets'][4]['motion'] for r in rows])
    np.testing.assert_array_equal(truth,previous['physical_targets'])
    predictions={arm:np.concatenate(values) for arm,values in predictions.items()}
    np.testing.assert_allclose(predictions['parent_action'],previous['predictions']['action'],rtol=0,atol=1e-6)
    for name in ('command_history','zero_motion','persistence','observed_future'):
        predictions[name]=np.asarray(previous['predictions'][name])
    # Verify actual applied suffixes independently of the readout labels.
    for directory,_,expected in prepared:
        with np.load(directory/'policy_histories.npz',allow_pickle=False) as a:
            np.testing.assert_allclose(a['applied_command_values'][18,-5:][:,[0,2]].reshape(10),expected,rtol=0,atol=1e-6)
    summaries={role:{arm:metrics(value[[i for i,r in enumerate(rows) if r['data_role']==role]],
                                truth[[i for i,r in enumerate(rows) if r['data_role']==role]])
                     for arm,value in predictions.items()} for role in ('train','geometry_transfer')}
    # Same pre-existing diagnostic goals; all actual tied choices contribute.
    decisions={}
    for arm,value in predictions.items():
        regrets=[]
        for offset in range(0,len(rows),3):
            if rows[offset]['data_role']!='geometry_transfer':continue
            for angle in (-np.pi/4,0,np.pi/4):
                goal=.25*np.array([np.cos(angle),np.sin(angle)])
                score=endpoint_waypoint_scores(torch.tensor(value[offset:offset+3]),goal).numpy()
                physical=np.linalg.norm(goal[None]-truth[offset:offset+3,:2],axis=1)
                ties=np.flatnonzero(score==score.max());regrets.append(float(physical[ties].mean()-physical.min()))
        decisions[arm]=dict(mean_point_goal_regret_mm=float(np.mean(regrets))*1000,contexts=len(regrets))
    report=dict(status='COMPLETE',summaries=summaries,waypoint_decisions=decisions,
                predictions={a:v.tolist() for a,v in predictions.items()},physical_targets=truth.tolist(),
                predictor_fit_sha256=fit.digest(fit.RESULT),readout_fit_sha256=fit.digest(readout.OUTPUT/'result.json'),
                source_sha256={p:fit.digest(p) for p in (__file__,'lewm/dense_endpoint_navigation_development.py')},
                previous_parent_predictions_reproduced=True,reused_fixed_baselines=list(('command_history','zero_motion','persistence','observed_future')),
                wall_s=time.monotonic()-started,new_navigation=False,horizon_ms=500,
                limitations=['exposed 36-branch panel; 18 transfer branches','free-space motions repeated across geometries',
                             'one frozen motion head; oracle is diagnostic only','endpoint scores do not establish collision feasibility',
                             'no intermediate motion, contact prediction, or delayed-dispatch forecast beyond 500ms'])
    fit.save(RESULT,report);fit.save(OUTPUT/'result.json',report)
    print('ENDPOINT_NAVIGATION_EVALUATION',json.dumps(dict(transfer=summaries['geometry_transfer'],decisions=decisions)),flush=True)


if __name__=='__main__':main()
