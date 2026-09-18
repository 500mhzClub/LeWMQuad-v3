"""Fixed transfer comparison of both continued heads on retained stall branches."""
import json
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from lewm.dense_visual_motion_readout_development import pool_tokens
from scripts import collect_go2_dense_stall_turn_branches_development as branch
from scripts import train_go2_full_heading_readout_development as training
from scripts import train_go2_horizon_dense_predictor_development as fit


@torch.inference_mode()
def main():
    plan = branch.read(branch.OUTPUT/'plan.json')
    old = branch.read(branch.OUTPUT/'evaluation/result.json')
    trained = branch.read(training.OUTPUT/'result.json')
    assert trained['status']=='COMPLETE' and old['status']=='COMPLETE'
    output = training.OUTPUT/'stall_branch_evaluation';output.mkdir(exist_ok=False)
    save = lambda name,value:branch.save(output/name,value)
    save('plan.json',dict(collection_plan_sha256=branch.digest(branch.OUTPUT/'plan.json'),
        source_sha256=branch.digest(__file__),readout_sha256=trained['checkpoint_sha256'],
        frame=plan['frame'],actions=plan['actions'],horizons_ms=plan['horizons_ms'],
        primary_readout_horizon_ms=500,no_fit=True,no_navigation=True,
        fixed_heads=['original','old_data','mixed_data'],fixed_future_inputs=['action','no_future_action','observed_future'],
        one_exposed_context=True,automatic_navigation_promotion=False))
    started = time.monotonic()
    try:
        torch.set_num_threads(4)
        encoder = fit.parent.reference.encoders.VJepa21Arm();encoder.build(torch.device('cuda:0'),torch.float32)
        cache = {}
        def encode(path):
            key = branch.digest(path)
            if key not in cache:
                cache[key] = F.layer_norm(encoder.tokens(encoder.preprocess(str(path))[None].cuda()).float(),(1024,))[0]
            return cache[key]
        frame = plan['frame'];actions = plan['actions'];horizons = [h//100 for h in plan['horizons_ms']]
        context = torch.stack([encode(branch.REFERENCE/'native'/f'rgb_{f:04d}.png') for f in (frame-10,frame-5,frame)])[None]
        with np.load(branch.REFERENCE/'native/policy_histories.npz',allow_pickle=False) as archive:
            past = archive['applied_command_values'][frame].astype(np.float32)
        stats = branch.read(fit.parent.reference.CACHE/'proprio_v1/proprio_norm_stats.json')
        control = ((torch.from_numpy(past[:,[0,2]].reshape(3,5,2))-torch.tensor(stats['control_mean']))/torch.tensor(stats['control_std'])).float()[None].cuda()
        tape = torch.tensor([plan['branches'][a]['applied'] for a in actions],dtype=torch.float32,device='cuda')[:,:,[0,2]]
        mask = torch.ones(3,768,dtype=torch.bool,device='cuda');features = {}
        for arm in ('action','no_future_action'):
            model = fit.load(arm).cuda();features[arm] = {}
            for h in horizons:
                n = 1 if h<=3 or arm=='no_future_action' else 3
                z = model(context.expand(n,-1,-1,-1),tape[:n],torch.full((n,),h,dtype=torch.long,device='cuda'),mask[:n],control=control.expand(n,-1,-1,-1))
                features[arm][h] = F.layer_norm(z.float(),(1024,)).expand(3,-1,-1)
            del model
        features['observed_future'] = {h:torch.stack([encode(branch.OUTPUT/a/f'rgb_{frame+h:04d}.png') for a in actions]) for h in horizons}
        heads = dict(original=training.original.load().cuda())|{a:training.load(a).cuda() for a in training.ARMS}
        current = pool_tokens(context[:,-1]).expand(3,-1,-1)
        rows = []
        for h in horizons:
            outputs = {head:{arm:m(current,pool_tokens(values[h])).cpu().numpy() for arm,values in features.items()} for head,m in heads.items()}
            for i,action in enumerate(actions):
                previous = next(r for r in old['rows'] if r['action']==action and r['horizon_ms']==h*100)
                truth = np.asarray(previous['actual_xy_yaw'])
                predictions = {};errors = {}
                for head,arms in outputs.items():
                    predictions[head] = {};errors[head] = {}
                    for arm,values in arms.items():
                        prediction = values[i];delta = prediction-truth;delta[2] = np.arctan2(np.sin(delta[2]),np.cos(delta[2]))
                        predictions[head][arm] = prediction.tolist();errors[head][arm] = delta.tolist()
                        if head=='original':
                            np.testing.assert_allclose(prediction,previous['predictions'][arm],rtol=0,atol=2e-5)
                rows.append(dict(action=action,horizon_ms=h*100,actual_xy_yaw=truth.tolist(),predictions=predictions,errors=errors))
        metrics = {}
        for h in (500,700):
            selected = [r for r in rows if r['horizon_ms']==h];metrics[str(h)] = {}
            for head in heads:
                metrics[str(h)][head] = {}
                for arm in features:
                    error = np.asarray([r['errors'][head][arm] for r in selected])
                    metrics[str(h)][head][arm] = dict(xy_rmse_mm=float(1000*np.sqrt(np.mean(np.sum(error[:,:2]**2,axis=1)))),
                        yaw_rmse_deg=float(np.degrees(np.sqrt(np.mean(error[:,2]**2)))))
        increments = []
        for action in actions:
            a = next(r for r in rows if r['action']==action and r['horizon_ms']==300)
            b = next(r for r in rows if r['action']==action and r['horizon_ms']==700)
            increments.append(dict(action=action,actual=(np.asarray(b['actual_xy_yaw'])-a['actual_xy_yaw']).tolist(),
                predictions={head:{arm:(np.asarray(b['predictions'][head][arm])-a['predictions'][head][arm]).tolist() for arm in features} for head in heads}))
        result = dict(status='COMPLETE',rows=rows,metrics=metrics,commit_interval=increments,
            wall_s=time.monotonic()-started,original_outputs_reproduced_atol=2e-5,
            encoder_and_predictor_unchanged=True,no_navigation=True,
            scope='one exposed context; readout data intervention, not JEPA objective isolation or prospective navigation')
        save('result.json',result);print('FULL_HEADING_READOUT_TRANSFER',json.dumps(metrics),flush=True)
    except BaseException as error:
        save('failure.json',dict(reason=repr(error),traceback=traceback.format_exc()));raise


if __name__=='__main__':
    main()
