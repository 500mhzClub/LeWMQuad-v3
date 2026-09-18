"""Matched action/no-future-action predictors in one frozen visual target space."""
import argparse
import gc
import json
from pathlib import Path
import resource
import time

import cv2
import numpy as np
import torch

from lewm.anchored_visual_dynamics_development import AnchoredVisualDynamics
from lewm.frozen_representation_dynamics_development import per_context_loss
from lewm.pulse_timed_training_runner_development import state_digest
from scripts import train_go2_visual_target_jepa_development as representation
from scripts import train_go2_frozen_representation_dynamics_development as datafit
from scripts import probe_go2_jepa_latent_branch_science_development as probe

OUTPUT = probe.fits.BASE/'go2_anchored_visual_dynamics_v1_attempt_001'
PLAN = Path('docs/go2_anchored_visual_dynamics_plan_2026-09-17.json')
ARMS = ('action','no_future_action')


def prepare():
    assert not OUTPUT.exists()
    schedule=json.loads(datafit.SCHEDULE.read_text())
    probe.save(PLAN,dict(schema='anchored_visual_dynamics.v1',arms=ARMS,
        representation_fit_sha256=probe.digest(representation.OUTPUT/'fit.json'),
        schedule_sha256=probe.digest(datafit.SCHEDULE),seed=schedule['seed'],
        contexts=4694,draws=7200,updates=1200,batch_size=6,learning_rate=.001,
        anchor='frozen EMA visual representation of current RGB only',
        context='four frozen online multimodal observations, trainable history GRU',
        prediction='current visual state plus cumulative normalized innovations',
        initialization='identical weights; final transition layer zero gives exact persistence',
        target='same frozen visual target in both arms; no EMA or encoder changes',
        normalization='training draw-weighted RMS innovation per coordinate, floor 1e-4; no mean subtraction',
        loss='normalized latent MSE averaged per context over available future images',
        optimizer='AdamW, zero weight decay, gradient norm clipping 1',
        no_motion_loss=True,no_checkpoint_selection=True,final_update_only=True,
        evaluation='all fixed branch contexts; all horizons, 800ms transfer primary; persistence, no-action, target means',
        no_JEPA_training_superiority_inferred_from_predictor_fit=True,
        source_sha256={p:probe.digest(p) for p in (__file__,'lewm/anchored_visual_dynamics_development.py')},
        resources=dict(cpu_core=8,threads=1,ram_available_gib=72,output_free_gib=4.4,both_gpus_idle=True,
            strategy='encode once, release raw tensors, fit two small predictors sequentially with shared cache')))
    OUTPUT.mkdir();print('PREPARED anchored visual action/no-action comparison',flush=True)


@torch.no_grad()
def encode(model, rows, raw):
    result={k:[] for k in ('past','anchor','target','blocks','valid','available')}
    for start in range(0,len(rows),16):
        selected=rows[start:start+16];b=datafit.batch(raw,[r['sample_id'] for r in selected])
        inp=b['inputs'];tar=b['targets'];history=inp['observation_history']
        past=model.encoder({k:v.flatten(0,1) for k,v in history.items()}).reshape(len(selected),4,32)
        anchor=model.target({'rgb':history['rgb'][:,-1]})
        mask=tar['future_valid'];target=torch.zeros((len(selected),8,32))
        if mask.any():target[mask]=model.target({'rgb':tar['future_observations']['rgb'][mask]})
        values=dict(past=past,anchor=anchor,target=target,blocks=inp['known_action_blocks'],
                    valid=inp['known_action_valid'],available=mask)
        for k,v in values.items():result[k].append(v)
    return {k:torch.cat(v) for k,v in result.items()}


def fresh(statistics,seed,arm):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return AnchoredVisualDynamics(statistics,use_future_actions=arm=='action')


@torch.no_grad()
def training_error(model,cache,weights):
    errors=[];normalized=[];masks=[]
    for start in range(0,len(weights),64):
        s=slice(start,start+64)
        p=model(*(cache[k][s] for k in ('past','anchor','blocks','valid')))
        e,m=per_context_loss(p,cache['target'][s],cache['available'][s])
        n,_=per_context_loss((p-cache['target'][s])/model.innovation_scale,torch.zeros_like(p),cache['available'][s])
        errors.append(e);normalized.append(n);masks.append(m)
    mask=torch.cat(masks);w=weights*mask
    return dict(mse=float((torch.cat(errors)*w).sum()/w.sum()),
                normalized_mse=float((torch.cat(normalized)*w).sum()/w.sum()))


def fit():
    plan=json.loads(PLAN.read_text())
    for p,sha in plan['source_sha256'].items():assert probe.digest(p)==sha
    assert probe.digest(datafit.SCHEDULE)==plan['schedule_sha256']
    assert probe.digest(representation.OUTPUT/'fit.json')==plan['representation_fit_sha256']
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    probe.save(OUTPUT/'launch.json',dict(status='started'))
    started=time.monotonic();model=representation.load();fixed_sha=state_digest(model.state_dict())
    rows=datafit.data.load_training_rows();assert len(rows)==4694 and all(r['data_role']=='train' for r in rows)
    raw,identities=datafit.data.prepare(rows)
    assert identities==json.loads((representation.OUTPUT/'consumed_policy_sha256.json').read_text())
    cache=encode(model,rows,raw);del raw;gc.collect()
    assert state_digest(model.state_dict())==fixed_sha
    schedule=json.loads(datafit.SCHEDULE.read_text())
    weights=torch.tensor([schedule['context_draw_counts'][r['sample_id']] for r in rows],dtype=torch.float64)
    available=cache['available'];counts=available.sum(-1).clamp_min(1)
    w=weights[:,None]*available/counts[:,None];innovation=cache['target'].double()-cache['anchor'][:,None].double()
    scale=((innovation.square()*w[:,:,None]).sum((0,1))/w.sum()).sqrt().clamp_min(1e-4)
    mean=(cache['anchor'].double()*weights[:,None]).sum(0)/weights.sum()
    anchor_scale=(((cache['anchor'].double()-mean).square()*weights[:,None]).sum(0)/weights.sum()).sqrt().clamp_min(1e-4)
    statistics=dict(anchor_mean=mean.float(),anchor_scale=anchor_scale.float(),innovation_scale=scale.float())
    lookup={r['sample_id']:i for i,r in enumerate(rows)};records={};initial_hashes=[]
    for arm in ARMS:
        predictor=fresh(statistics,plan['seed'],arm);initial_hashes.append(state_digest(predictor.state_dict()))
        with torch.no_grad():
            initial=predictor(*(cache[k][:6] for k in ('past','anchor','blocks','valid')))
            expected=torch.where(cache['valid'][:6],cache['anchor'][:6,None].expand_as(initial),torch.zeros_like(initial))
            torch.testing.assert_close(initial,expected,rtol=0,atol=0)
        before=training_error(predictor,cache,weights)
        optimizer=torch.optim.AdamW(predictor.parameters(),lr=.001,weight_decay=0.)
        with (OUTPUT/f'{arm}_updates.jsonl').open('x') as ledger:
            for step,ids in enumerate(schedule['batches'],1):
                idx=[lookup[i] for i in ids];optimizer.zero_grad(set_to_none=True)
                pred=predictor(*(cache[k][idx] for k in ('past','anchor','blocks','valid')))
                # Scale the difference, avoiding cancellation from dividing two large anchors.
                errors,mask=per_context_loss((pred-cache['target'][idx])/predictor.innovation_scale,
                    torch.zeros_like(pred),cache['available'][idx])
                loss=errors[mask].mean();assert torch.isfinite(loss);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(predictor.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();ledger.write(json.dumps(dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm)))+'\n')
                if step%300==0:ledger.flush();print('ANCHORED_VISUAL_FIT',arm,step,round(time.monotonic()-started,1),flush=True)
        assert step==1200
        after=training_error(predictor,cache,weights)
        checkpoint=OUTPUT/f'{arm}.pt'
        with checkpoint.open('xb') as stream:torch.save(dict(statistics=statistics,state=predictor.state_dict(),arm=arm),stream)
        clone=fresh(statistics,plan['seed'],arm);clone.load_state_dict(torch.load(checkpoint,weights_only=True)['state'])
        with torch.no_grad():
            args=tuple(cache[k][:6] for k in ('past','anchor','blocks','valid'))
            torch.testing.assert_close(clone(*args),predictor(*args),rtol=0,atol=0)
        records[arm]=dict(status='complete',before=before,after=after,updates=step,
                          checkpoint_sha256=probe.digest(checkpoint),reload_exact=True)
        print('ANCHORED_VISUAL_ARM_COMPLETE',arm,before,after,flush=True)
    assert len(set(initial_hashes))==1
    probe.save(OUTPUT/'result.json',dict(status='complete',records=records,initial_state_sha256=initial_hashes[0],
        fixed_representation_sha256=fixed_sha,representation_unchanged=True,inputs_match_original_training=True,
        initial_predictions_exact_persistence=True,normalization_training_only=True,
        wall_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


def load(arm):
    records=json.loads((OUTPUT/'result.json').read_text())['records']
    path=OUTPUT/f'{arm}.pt';assert probe.digest(path)==records[arm]['checkpoint_sha256']
    value=torch.load(path,map_location='cpu',weights_only=True)
    model=fresh(value['statistics'],0,arm);model.load_state_dict(value['state'])
    return model.eval().requires_grad_(False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args();prepare() if args.prepare else fit()
