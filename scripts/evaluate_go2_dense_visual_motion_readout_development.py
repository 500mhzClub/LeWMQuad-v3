"""Common motion probe on observed/predicted futures and established controls.

All 36 original branches, same 500-ms horizon. No fitting or navigation. Future
RGB and physical targets are used only after causal predictions are complete.
"""
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from lewm.dense_visual_motion_readout_development import pool_tokens
from scripts import train_go2_dense_visual_motion_readout_development as training

parent=training.parent
reference=parent.reference
OUTPUT=training.OUTPUT/'branch_evaluation'
RESULT=Path('docs/go2_dense_visual_motion_readout_branch_result_2026-09-17.json')


@torch.inference_mode()
def command_reference(rows):
    from lewm.pulse_timed_dataset_development import stack_samples
    from scripts import fit_go2_frozen_motion_readout_development as previous
    model=previous.load_parent('jepa',json.loads(previous.PLAN.read_text())).eval()
    values=[]
    for row in rows:
        directory=reference.data.PULSE/row['trial'];past=row['history_observation_indices']
        packets={i:previous.source.load_route_observation(directory,i) for i in past}
        inputs=stack_samples([previous.source.inputs(row,previous.source.PacketReader(packets,past))])
        value=model.reference_motion(inputs['observation_history'],inputs['known_action_blocks'],inputs['known_action_valid'])
        values.append(value[0,4].cpu().numpy())
    return np.asarray(values)


def metrics(prediction,target):
    error=np.asarray(prediction,float)-np.asarray(target,float)
    yaw=np.arctan2(np.sin(error[:,2]),np.cos(error[:,2]))
    return dict(contexts=len(error),xy_rmse_mm=float(np.sqrt(np.mean(np.sum(error[:,:2]**2,axis=1)))*1000),
        yaw_rmse_deg=float(np.sqrt(np.mean(yaw*yaw))*180/np.pi))


@torch.inference_mode()
def run():
    fitted=json.loads((training.OUTPUT/'result.json').read_text())
    assert fitted['status']=='COMPLETE' and fitted['epochs']==training.EPOCHS
    OUTPUT.mkdir(exist_ok=False);started=time.monotonic();torch.set_num_threads(4)
    head=training.load().cuda()
    dynamics={};terminal=json.loads((parent.OUTPUT/'result.json').read_text())
    for arm in parent.ARMS:
        path=parent.OUTPUT/f'{arm}_latest.pt'
        assert reference.digest(path)==terminal['checkpoint_sha256'][arm]
        state=torch.load(path,map_location='cpu',weights_only=False)
        model=reference.ProprioActionPredictor(use_proprio=False)
        model.load_state_dict(state['model_state_dict'],strict=True)
        dynamics[arm]=model.cuda().eval().requires_grad_(False)
        del state
    rows=reference.selected_rows();prepared=[reference.inputs(r) for r in rows]
    stats=json.loads((reference.CACHE/'proprio_v1/proprio_norm_stats.json').read_text())
    mean,std=(np.asarray(stats[k],np.float32) for k in ('control_mean','control_std'))
    encoder=reference.encoders.VJepa21Arm();encoder.build(torch.device('cuda:0'),torch.float32)
    cache={}

    def encode(path):
        key=reference.digest(path)
        if key not in cache:
            pixel=encoder.preprocess(str(path))[None].cuda()
            cache[key]=F.layer_norm(encoder.tokens(pixel).float(),(1024,))[0].cpu()
        return cache[key]

    contexts=torch.stack([torch.stack([encode(d/f'rgb_{i:04d}.png') for i in (3,8,13)]) for d,_,_ in prepared])
    controls=torch.from_numpy(np.stack([(c-mean)/std for _,c,_ in prepared]))
    actions=torch.from_numpy(np.stack([a for _,_,a in prepared]))
    predictions={a:[] for a in (*parent.ARMS,'persistence')}
    for offset in range(0,len(rows),3):
        x,a,c=(v[offset:offset+3].cuda() for v in (contexts,actions,controls))
        assert torch.equal(x,x[:1].expand_as(x)) and torch.equal(c,c[:1].expand_as(c))
        current=pool_tokens(x[:,-1]);mask=torch.ones(3,768,dtype=torch.bool,device='cuda')
        for arm,model in dynamics.items():
            if arm=='no_future_action':
                future=F.layer_norm(model(x[:1],torch.zeros_like(a[:1]),mask[:1],control=c[:1]).float(),(1024,))
                value=head(current[:1],pool_tokens(future)).expand(3,-1)
            else:
                future=F.layer_norm(model(x,a,mask,control=c).float(),(1024,))
                value=head(current,pool_tokens(future))
            predictions[arm].append(value.cpu().numpy())
        predictions['persistence'].append(head(current[:1],current[:1]).expand(3,-1).cpu().numpy())
    predictions={a:np.concatenate(v) for a,v in predictions.items()}
    predictions['command_history']=command_reference(rows)
    predictions['zero_motion']=np.zeros((len(rows),3))
    reference.save(OUTPUT/'causal_predictions_complete.json',dict(contexts=len(rows),future_rgb_loaded=False))
    # Only the oracle diagnostic now receives actual future visual features.
    observed=[];truth=[]
    for i,(row,(directory,_,expected)) in enumerate(zip(rows,prepared,strict=True)):
        target=row['targets'][4]
        assert target['motion_valid'] and target['offset_ns']==500_000_000 and target['future_observation_index']==18
        with np.load(directory/'policy_histories.npz',allow_pickle=False) as archive:
            np.testing.assert_allclose(archive['applied_command_values'][18,-5:][:,[0,2]].reshape(10),expected,rtol=0,atol=1e-6)
        future=encode(directory/'rgb_0018.png')[None].cuda()
        current=contexts[i,-1:,:,:].cuda()
        observed.append(head(pool_tokens(current),pool_tokens(future))[0].cpu().numpy())
        truth.append(target['motion'])
    predictions['observed_future']=np.asarray(observed);truth=np.asarray(truth,float)
    assert all(np.isfinite(value).all() for value in predictions.values())
    summaries={};grouped={}
    for i,row in enumerate(rows): grouped.setdefault((row['data_role'],row['cluster'],row['prefix_action']),[]).append(i)
    groups=[]
    for key,indices in grouped.items():
        assert len(indices)==3
        centered_truth=truth[indices]-truth[indices].mean(0)
        groups.append(dict(role=key[0],cluster=key[1],prefix_action=key[2],trials=[rows[i]['trial'] for i in indices],
            models={a:dict(factual=metrics(v[indices],truth[indices]),
                centered_effect=metrics(v[indices]-v[indices].mean(0),centered_truth)) for a,v in predictions.items()}))
    for role in ('train','geometry_transfer'):
        indices=[i for i,row in enumerate(rows) if row['data_role']==role]
        summaries[role]={a:metrics(v[indices],truth[indices]) for a,v in predictions.items()}
    result=dict(status='COMPLETE',summaries=summaries,groups=groups,
        trials=[r['trial'] for r in rows],physical_targets=truth.tolist(),
        predictions={a:v.tolist() for a,v in predictions.items()},
        source_sha256=reference.digest(__file__),fit_sha256=reference.digest(training.OUTPUT/'result.json'),
        predictor_fit_sha256=reference.digest(parent.OUTPUT/'result.json'),
        head_shared_across_all_visual_arms=True,neural_readout_has_no_direct_action_inputs=True,
        head_trained_on_true_future_only=True,observed_future_is_oracle_not_deployable=True,
        horizon_ms=500,wall_s=time.monotonic()-started,new_navigation=False,
        limitations=['two exposed transfer geometries; six history groups per role',
            'one readout architecture/seed; poor decoding does not establish missing information',
            'probe input distribution differs between true and predicted future features',
            'existing command baseline used its original larger training population and four command histories',
            'motion endpoint only; no collision, clearance or eight-horizon runtime output'])
    reference.save(OUTPUT/'result.json',result);reference.save(RESULT,result)
    print('DENSE_MOTION_BRANCH_COMPLETE',json.dumps(summaries),flush=True)


if __name__=='__main__':
    try: run()
    except Exception as error:
        if OUTPUT.exists() and not (OUTPUT/'failure.json').exists():
            reference.save(OUTPUT/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc()))
        raise
