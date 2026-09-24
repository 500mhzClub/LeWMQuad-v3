"""Matched continuation: dense L1 versus frozen goal-embedding auxiliary loss."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from lewm.dense_visual_motion_readout_development import pool_tokens
from scripts import train_go2_frozen_vjepa_native_adaptation_development as parent
from scripts import train_go2_dense_goal_metric_development as metric_fit

OUTPUT = parent.OUTPUT.parent/'go2_dense_task_predictor_v1_attempt_001'
PLAN = Path('docs/go2_dense_task_predictor_plan_2026-09-17.json')
RESULT = Path('docs/go2_dense_task_predictor_fit_result_2026-09-17.json')
SEED, EPOCHS, BATCH = 2026091708, 8, 16
ARMS = {
    'dense_action': ('action', False),
    'metric_action': ('action', True),
    'dense_no_future_action': ('no_future_action', False),
    'metric_no_future_action': ('no_future_action', True),
}
save, digest = parent.save, parent.digest


def prepare():
    assert not OUTPUT.exists() and not PLAN.exists()
    terminal = json.loads((parent.OUTPUT/'result.json').read_text())
    metric = json.loads((metric_fit.OUTPUT/'result.json').read_text())
    assert terminal['status'] == metric['status'] == 'COMPLETE'
    samples = json.loads((parent.OUTPUT/'samples.json').read_text())
    admitted = {r['sample_id']:r for r in parent.load_training_rows()}
    assert len(samples) == 3518 and all(admitted[s['sample_id']]['data_role'] == 'train' for s in samples)
    initial = {a:digest(parent.OUTPUT/f'{a}_latest.pt') for a in parent.ARMS}
    assert initial == terminal['checkpoint_sha256']
    assert digest(metric_fit.OUTPUT/'metric.pt') == metric['checkpoint_sha256']
    sizes = {a:(parent.OUTPUT/f'{a}_latest.pt').stat().st_size for a in parent.ARMS}
    reserve = 512*1024**2
    assert shutil.disk_usage(OUTPUT.parent).free > 5*max(sizes.values())+reserve
    plan = dict(schema='dense_task_predictor.v1', arms=ARMS, seed=SEED, epochs=EPOCHS,
        samples=len(samples), effective_batch=BATCH, microbatch=BATCH, encoder_batch=8,
        initial_checkpoint_sha256=initial, metric_checkpoint_sha256=metric['checkpoint_sha256'],
        samples_sha256=digest(parent.OUTPUT/'samples.json'),
        frame_paths_sha256=digest(parent.OUTPUT/'frame_paths.json'),
        source_sha256={p:digest(p) for p in (__file__, 'lewm/dense_goal_metric_development.py',
            'lewm/dense_visual_motion_readout_development.py',
            'scripts/train_go2_frozen_vjepa_native_adaptation_development.py')},
        objective='dense L1 + lambda * mean(log1p(mean squared frozen goal-embedding error))',
        coefficient='initial dense L1 / auxiliary loss, averaged on 128 seeded training samples using action parent; fixed thereafter and shared by both auxiliary arms',
        calibration_samples=128, calibration_uses_transfer=False,
        continuation='restore parent model and AdamW states; same extra updates and example order in all arms',
        optimizer=dict(name='AdamW',lr=.0003,weight_decay=.01,gradient_clip=1.),
        encoder_frozen=True, metric_frozen=True, future_visual_target_only=True,
        extra_physical_supervision='only inherited frozen goal metric; no new pose or control target',
        checkpoint_rule='fixed final extra epoch; no transfer selection',
        precision='float32 encoder/head; bf16 predictor autocast with float32 losses',
        feature_storage='full normalized tokens, FP16 host RAM only; identical RGB reuse',
        concurrency='one GPU process; four arms interleaved, identical minibatches; CPU cores 8-11',
        resources=dict(available_ram_kib=next(int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')),
            output_free_bytes=shutil.disk_usage(OUTPUT.parent).free,
            gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory),
        storage='four latest resumable checkpoints; at most one temporary replacement; 512 MiB free reserve',
        evaluation='fixed exposed branch fidelity, earlier near-goal and late-turn rankings, then same prospective local reaching tasks; retain failures',
        limitations=['single continuation seed', 'auxiliary objective inherits physical supervision',
            'exposed development diagnostics; no isolated JEPA representation benefit or final maze evaluation'])
    OUTPUT.mkdir(); save(PLAN,plan); save(OUTPUT/'plan.json',plan)
    print('TASK_PREDICTOR_PREPARED',json.dumps(dict(arms=list(ARMS),epochs=EPOCHS,samples=len(samples))),flush=True)


@torch.no_grad()
def encode(paths):
    encoder = parent.reference.encoders.VJepa21Arm()
    encoder.build(torch.device('cuda:0'),torch.float32)
    keys = [digest(p) for p in paths]; first = {}
    for i,key in enumerate(keys): first.setdefault(key,i)
    unique = list(first.values()); cache = torch.empty(len(paths),768,1024,dtype=torch.float16)
    started = time.monotonic(); discrepancy = None
    for start in range(0,len(unique),8):
        indices = unique[start:start+8]
        pixels = torch.stack([encoder.preprocess(str(paths[i])) for i in indices]).cuda()
        features = F.layer_norm(encoder.tokens(pixels).float(),(1024,))
        assert features.shape == (len(indices),768,1024) and torch.isfinite(features).all()
        if start == 0:
            one = F.layer_norm(encoder.tokens(pixels[:1]).float(),(1024,))
            discrepancy = float((features[:1]-one).square().mean()); assert discrepancy < 1e-8
        cache[indices] = features.cpu().half()
        if start//8 % 25 == 0 or start+8 >= len(unique):
            print('TASK_FEATURES',min(start+8,len(unique)),len(unique),'seconds',round(time.monotonic()-started,1),flush=True)
    for i,key in enumerate(keys):
        if i != first[key]: cache[i].copy_(cache[first[key]])
    del encoder; torch.cuda.empty_cache()
    return cache,dict(paths=len(paths),unique=len(unique),seconds=time.monotonic()-started,
        first_image_batch_vs_single_mse=discrepancy,cache_bytes=cache.numel()*cache.element_size())


def prediction(model,x,a,c):
    mask = torch.ones(len(x),768,dtype=torch.bool,device=x.device)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        value = model(x,a,mask,control=c)
    return F.layer_norm(value.float(),(1024,))


def losses(model,head,x,a,c,y,target_embedding):
    pred = prediction(model,x,a,c)
    dense = F.l1_loss(pred,y)
    embedding = head.embed(pool_tokens(pred))
    auxiliary = torch.log1p((embedding-target_embedding).square().mean(-1)).mean()
    return dense,auxiliary


def load(arm):
    result = json.loads(RESULT.read_text()); assert result['status'] == 'COMPLETE'
    path = OUTPUT/f'{arm}_latest.pt'; assert digest(path) == result['checkpoint_sha256'][arm]
    state = torch.load(path,map_location='cpu',weights_only=False); assert state['epoch'] == EPOCHS-1
    model = parent.reference.ProprioActionPredictor(use_proprio=False)
    model.load_state_dict(state['model_state_dict'],strict=True)
    return model.eval().requires_grad_(False)


def fit():
    plan = json.loads(PLAN.read_text())
    assert all(digest(p) == h for p,h in plan['source_sha256'].items())
    assert digest(parent.OUTPUT/'samples.json') == plan['samples_sha256']
    assert digest(parent.OUTPUT/'frame_paths.json') == plan['frame_paths_sha256']
    assert digest(metric_fit.OUTPUT/'metric.pt') == plan['metric_checkpoint_sha256']
    assert not (OUTPUT/'progress.jsonl').exists() and not (OUTPUT/'process.json').exists()
    save(OUTPUT/'process.json',dict(pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),plan_sha256=digest(PLAN)))
    torch.set_num_threads(4); torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    started = time.monotonic()
    samples = json.loads((parent.OUTPUT/'samples.json').read_text())
    paths = [Path(p) for p in json.loads((parent.OUTPUT/'frame_paths.json').read_text())]
    features,encoding = encode(paths); save(OUTPUT/'encoding.json',encoding)
    indices = torch.tensor([s['frames'] for s in samples])
    actions = torch.tensor([s['action'] for s in samples],dtype=torch.float32)
    controls = torch.tensor([s['control'] for s in samples],dtype=torch.float32)
    models,optimizers = {},{}
    for arm,(initial,_) in ARMS.items():
        path = parent.OUTPUT/f'{initial}_latest.pt'; assert digest(path) == plan['initial_checkpoint_sha256'][initial]
        state = torch.load(path,map_location='cpu',weights_only=False); assert state['epoch'] == 23
        model = parent.reference.ProprioActionPredictor(use_proprio=False).cuda().train()
        model.load_state_dict(state['model_state_dict'],strict=True)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        models[arm],optimizers[arm] = model,optimizer
        del state
    for initial in parent.ARMS:
        a,b = (models[f'{kind}_{initial}'] for kind in ('dense','metric'))
        assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters(),strict=True))
    head = metric_fit.load().cuda(); assert not any(p.requires_grad for p in head.parameters())
    order = torch.randperm(len(samples),generator=torch.Generator().manual_seed(SEED-1))[:128]
    dense_total = auxiliary_total = 0.
    with torch.no_grad():
        for chunk in order.split(BATCH):
            v = features[indices[chunk]].float().cuda(); x,y = v[:,:3],v[:,3]
            target = head.embed(pool_tokens(y))
            dense,aux = losses(models['dense_action'],head,x,actions[chunk].cuda(),controls[chunk].cuda(),y,target)
            dense_total += float(dense)*len(chunk); auxiliary_total += float(aux)*len(chunk)
    coefficient = dense_total/auxiliary_total
    assert np.isfinite(coefficient) and coefficient > 0
    save(OUTPUT/'calibration.json',dict(samples=order.tolist(),dense_l1=dense_total/128,
        log_embedding_mse=auxiliary_total/128,coefficient=coefficient,transfer_used=False))
    print('TASK_LOSS_CALIBRATED',coefficient,flush=True)
    histories = []; updates = 0; torch.cuda.reset_peak_memory_stats()
    with (OUTPUT/'progress.jsonl').open('x') as progress:
        for epoch in range(EPOCHS):
            epoch_started = time.monotonic(); totals = {arm:np.zeros(3) for arm in ARMS}
            order = torch.randperm(len(samples),generator=torch.Generator().manual_seed(SEED+epoch))
            for chunk in order.split(BATCH):
                v = features[indices[chunk]].float().cuda(); x,y = v[:,:3],v[:,3]
                a,c = actions[chunk].cuda(),controls[chunk].cuda()
                with torch.no_grad(): target = head.embed(pool_tokens(y))
                cpu_rng,cuda_rng = torch.get_rng_state(),torch.cuda.get_rng_state()
                for arm,(initial,use_metric) in ARMS.items():
                    torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
                    optimizer = optimizers[arm]; optimizer.zero_grad(set_to_none=True)
                    command = a if initial == 'action' else torch.zeros_like(a)
                    dense,aux = losses(models[arm],head,x,command,c,y,target)
                    total = dense+coefficient*aux if use_metric else dense
                    assert torch.isfinite(total) and torch.isfinite(aux)
                    total.backward(); grad = torch.nn.utils.clip_grad_norm_(models[arm].parameters(),1.)
                    assert torch.isfinite(grad) and all(p.grad is None for p in head.parameters())
                    optimizer.step()
                    totals[arm] += np.array([float(dense.detach()),float(aux.detach()),float(total.detach())])*len(chunk)
                updates += 1
                if updates % 50 == 0 or updates == 1:
                    print('TASK_UPDATES',updates,'epoch',epoch,'seconds',round(time.monotonic()-started,1),
                        'peak_gpu_gib',round(torch.cuda.max_memory_allocated()/1024**3,2),flush=True)
            for arm in ARMS:
                assert shutil.disk_usage(OUTPUT).free > 768*1024**2, 'checkpoint reserve exhausted'
                parent.checkpoint(OUTPUT/f'{arm}_latest.pt',models[arm],optimizers[arm],epoch,plan)
            row = dict(epoch=epoch,updates_per_arm=updates,epoch_s=time.monotonic()-epoch_started,
                elapsed_s=time.monotonic()-started,
                losses={arm:dict(zip(('dense_l1','log_embedding_mse','objective'),(v/len(samples)).tolist())) for arm,v in totals.items()})
            histories.append(row); progress.write(json.dumps(row)+'\n'); progress.flush()
            print('TASK_EPOCH',json.dumps(row),flush=True)
    result = dict(status='COMPLETE',epochs=EPOCHS,updates_per_arm=updates,coefficient=coefficient,
        histories=histories,encoding=encoding,wall_s=time.monotonic()-started,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        checkpoint_sha256={a:digest(OUTPUT/f'{a}_latest.pt') for a in ARMS},
        encoder_frozen=True,metric_frozen=True,no_transfer_data_used=True,navigation_tested=False)
    save(OUTPUT/'result.json',result); save(RESULT,result)
    print('TASK_PREDICTOR_COMPLETE',json.dumps({k:v for k,v in result.items() if k!='histories'}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true'); args = parser.parse_args()
    try:
        prepare() if args.prepare else fit()
    except Exception as error:
        if OUTPUT.exists():
            path = OUTPUT/('prepare_failure.json' if args.prepare else 'fit_failure.json')
            if not path.exists(): save(path,dict(reason=repr(error),traceback=traceback.format_exc()))
        raise
