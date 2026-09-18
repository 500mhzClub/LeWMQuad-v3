"""Fixed final-checkpoint comparison on the previously exposed native branches."""
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from scripts import evaluate_go2_frozen_vjepa_native_branches_development as baseline
from scripts import train_go2_frozen_vjepa_native_adaptation_development as training

OUTPUT = training.OUTPUT/'branch_evaluation_attempt_002'
RESULT = Path('docs/go2_frozen_vjepa_native_adaptation_branch_result_2026-09-17.json')


@torch.inference_mode()
def run():
    terminal=json.loads((training.OUTPUT/'result.json').read_text())
    assert terminal['status']=='COMPLETE' and terminal['epochs']==training.EPOCHS
    assert not OUTPUT.exists()
    OUTPUT.mkdir()
    started=time.monotonic();torch.set_num_threads(4)
    models={}
    for arm in training.ARMS:
        path=training.OUTPUT/f'{arm}_latest.pt'
        assert baseline.digest(path)==terminal['checkpoint_sha256'][arm]
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['epoch']==training.EPOCHS-1
        m=baseline.ProprioActionPredictor(use_proprio=False)
        m.load_state_dict(state['model_state_dict'],strict=True)
        models[arm]=m.cuda().eval().requires_grad_(False)
        del state
    rows=baseline.selected_rows()
    prepared=[baseline.inputs(row) for row in rows]
    stats=json.loads((baseline.CACHE/'proprio_v1/proprio_norm_stats.json').read_text())
    mean,std=(np.asarray(stats[k],np.float32) for k in ('control_mean','control_std'))
    encoder=baseline.encoders.VJepa21Arm()
    encoder.build(torch.device('cuda:0'),torch.float32)
    cache={}

    def encode(path):
        key=baseline.digest(path)
        if key not in cache:
            pixels=encoder.preprocess(str(path))[None].cuda()
            cache[key]=F.layer_norm(encoder.tokens(pixels).float(),(1024,))[0].cpu()
        return cache[key]

    contexts=torch.stack([torch.stack([encode(d/f'rgb_{f:04d}.png') for f in (3,8,13)])
                          for d,_,_ in prepared])
    control=torch.from_numpy(np.stack([(c-mean)/std for _,c,_ in prepared]))
    actions=torch.from_numpy(np.stack([a for _,_,a in prepared]))
    predictions={a:[] for a in training.ARMS}
    for offset in range(0,len(rows),3):
        x,a,c=(v[offset:offset+3].cuda() for v in (contexts,actions,control))
        mask=torch.ones(len(x),768,dtype=torch.bool,device='cuda')
        for arm,m in models.items():
            if arm=='no_future_action':
                assert torch.equal(x,x[:1].expand_as(x)) and torch.equal(c,c[:1].expand_as(c))
                # Identical effective inputs have one mathematical forecast.
                # Batched GPU roundoff must not manufacture action sensitivity.
                value=F.layer_norm(m(x[:1],torch.zeros_like(a[:1]),mask[:1],control=c[:1]).float(),(1024,))
                predictions[arm].append(value.expand(len(x),-1,-1).clone().cpu())
            else:
                predictions[arm].append(F.layer_norm(m(x,a,mask,control=c).float(),(1024,)).cpu())
    predictions={a:torch.cat(p) for a,p in predictions.items()}
    baseline.save(OUTPUT/'causal_forecasts_complete.json',dict(contexts=len(rows),future_rgb_loaded=False))
    targets=[]
    for d,_,expected in prepared:
        with np.load(d/'policy_histories.npz',allow_pickle=False) as archive:
            executed=archive['applied_command_values'][18][-5:,[0,2]].reshape(10)
        np.testing.assert_allclose(executed,expected,atol=1e-6,rtol=0)
        targets.append(encode(d/'rgb_0018.png'))
    target=torch.stack(targets)
    predictions['persistence']=contexts[:,-1]
    grouped={}
    for i,row in enumerate(rows):
        grouped.setdefault((row['data_role'],row['cluster'],row['prefix_action']),[]).append(i)
    details=[]
    for key,indices in grouped.items():
        assert len(indices)==3
        truth=target[indices]
        for i in indices[1:]:
            assert torch.equal(contexts[indices[0]],contexts[i])
            assert torch.equal(control[indices[0]],control[i])
        blind=predictions['no_future_action'][indices]
        assert torch.equal(blind[0],blind[1]) and torch.equal(blind[0],blind[2])
        record=dict(role=key[0],cluster=key[1],prefix_action=key[2],
                    trials=[rows[i]['trial'] for i in indices],models={})
        for arm,values in predictions.items():
            p=values[indices]
            dist=(p[:,None]-truth[None]).square().mean((-1,-2))
            wins=[bool(dist[j,j]<torch.cat((dist[:j,j],dist[j+1:,j])).min()) for j in range(3)]
            effect=((p-p.mean(0))-(truth-truth.mean(0))).square().mean()
            zero=(truth-truth.mean(0)).square().mean()
            record['models'][arm]=dict(factual_mse=dist.diag().tolist(),mse_matrix=dist.tolist(),
                correct_action_wins=wins,centered_effect_mse=float(effect),centered_zero_mse=float(zero))
        details.append(record)
    summaries={}
    for role in ('train','geometry_transfer'):
        groups=[g for g in details if g['role']==role]
        persistence=float(np.mean([g['models']['persistence']['factual_mse'] for g in groups]))
        summaries[role]={}
        for arm in predictions:
            records=[g['models'][arm] for g in groups]
            mse=float(np.mean([r['factual_mse'] for r in records]))
            summaries[role][arm]=dict(contexts=3*len(groups),mse=mse,mse_over_persistence=mse/persistence,
                correct_action_wins=sum(sum(r['correct_action_wins']) for r in records),
                centered_effect_ratio=float(np.mean([r['centered_effect_mse'] for r in records])/
                                            np.mean([r['centered_zero_mse'] for r in records])))
    previous=json.loads(baseline.RESULT.read_text())
    for role in summaries:
        np.testing.assert_allclose(summaries[role]['persistence']['mse'],
                                   previous['summaries'][role]['persistence']['mse'],rtol=0,atol=1e-6)
    result=dict(status='COMPLETE',summaries=summaries,groups=details,
        unchanged_checkpoint_reference=previous['summaries'],fit_sha256=baseline.digest(training.OUTPUT/'result.json'),
        fixed_epoch=training.EPOCHS-1,encoder_frozen=True,wall_s=time.monotonic()-started,
        navigation_tested=False,transfer_geometries_previously_exposed=True,
        training_role_scores_are_in_sample=True,no_dense_cache_retained=True,
        action_blind_single_forecast_per_identical_input_group=True,
        prior_evaluation_failure='branch_evaluation/failure.json',
        roundoff_diagnostic='branch_evaluation/identical_input_numerics.json')
    baseline.save(OUTPUT/'result.json',result);baseline.save(RESULT,result)
    print('NATIVE_ADAPTATION_BRANCH_COMPLETE',json.dumps(summaries),flush=True)


if __name__=='__main__':
    try:run()
    except Exception as error:
        if OUTPUT.exists() and not (OUTPUT/'failure.json').exists():
            baseline.save(OUTPUT/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc()))
        raise
