"""Matched 2x2 continuation: old/mixed data x future-action/action-blind."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch
import yaml

from lewm.geometry_progress_pilot_development import candidate_commands
from lewm_genesis.lewm_contract import SafetyLimits, apply_safety_limits_single
from scripts import collect_go2_balanced_start_actions_development as collection
from scripts import train_go2_frozen_vjepa_native_adaptation_development as parent
from scripts.train_go2_dense_task_predictor_development import encode

OUTPUT=collection.OUTPUT.parent/'go2_balanced_start_predictor_v1_attempt_001'
PLAN=Path('docs/go2_balanced_start_predictor_plan_2026-09-17.json')
RESULT=Path('docs/go2_balanced_start_predictor_fit_result_2026-09-17.json')
SEED,EPOCHS,BATCH=2026091710,8,16
ARMS={f'{data}_{arm}':(data,arm) for data in ('old','mixed') for arm in parent.ARMS}
save,digest=parent.save,parent.digest


def dataset():
    samples=json.loads((parent.OUTPUT/'samples.json').read_text())
    paths=json.loads((parent.OUTPUT/'frame_paths.json').read_text())
    original=len(samples);assert original==3518
    stats=json.loads((parent.reference.CACHE/'proprio_v1/proprio_norm_stats.json').read_text())
    mean,std=(np.asarray(stats[k],np.float32) for k in ('control_mean','control_std'))
    limits=SafetyLimits.from_manifest(yaml.safe_load(Path('config/go2_platform_manifest.yaml').read_text()))
    terminal=json.loads(collection.RESULT.read_text())
    assert terminal['status']=='COMPLETE' and terminal['eligible']==48 and terminal['contacts']==0
    for case,(trial,action) in enumerate(collection.CASES):
        root=collection.OUTPUT/f'case_{case:02d}'
        spec=json.loads((root/'specification.json').read_text());assert spec['data_role']=='train'
        with np.load(root/'policy_histories.npz',allow_pickle=False) as a:
            past=a['applied_command_values'][10].astype(np.float32)
            future=a['applied_command_values'][15][-5:].astype(np.float32)
            assert a['applied_command_valid'][10].all() and np.max(np.abs(past))<1e-6
            meta=json.loads((root/'policy_observations.json').read_text())
            times=[meta['frames'][i]['image_ns'] for i in (0,5,10,15)]
            assert np.diff(times).tolist()==[500_000_000]*3
            assert a['applied_command_measured_ns'][10][[4,9,14]].tolist()==times[:3]
            assert (a['applied_command_available_ns'][10]<=times[2]).all()
        expected=np.asarray(apply_safety_limits_single([candidate_commands(action)[0]]*5,tuple(past[-1]),limits)[0])
        np.testing.assert_allclose(future,expected,rtol=0,atol=1e-6)
        indices=list(range(len(paths),len(paths)+4))
        paths.extend(str(root/f'rgb_{i:04d}.png') for i in (0,5,10,15))
        samples.append(dict(sample_id=f'balanced_start/case_{case:02d}/frame_10',source='balanced_start',
                            trial=trial,frame=10,frames=indices,action=future[:,[0,2]].reshape(10).tolist(),
                            control=((past[:,[0,2]].reshape(3,5,2)-mean)/std).tolist()))
    return samples,paths,original


def schedule(original,new):
    rng=np.random.default_rng(SEED)
    queue=[];epochs=[];retained=set();counts=np.zeros(new,dtype=int)
    for epoch in range(EPOCHS):
        order=torch.randperm(original,generator=torch.Generator().manual_seed(SEED+epoch)).tolist()
        batches=[]
        for offset in range(0,original,BATCH):
            old=order[offset:offset+BATCH];mixed=list(old)
            positions=rng.choice(len(old),size=2,replace=False)
            for position in positions:
                if not queue:queue=rng.permutation(new).tolist()
                i=queue.pop();mixed[int(position)]=original+i;counts[i]+=1
            retained.update(i for i in mixed if i<original)
            batches.append(dict(old=old,mixed=mixed))
        epochs.append(batches)
    assert len(retained)==original and counts.max()-counts.min()<=1
    return epochs,counts.tolist()


def prepare():
    assert not OUTPUT.exists() and not PLAN.exists()
    samples,paths,original=dataset();epochs,counts=schedule(original,48)
    free=shutil.disk_usage(OUTPUT.parent).free
    assert free>(512+350)*1024**2,free
    terminal=json.loads((parent.OUTPUT/'result.json').read_text());assert terminal['status']=='COMPLETE'
    initial={a:digest(parent.OUTPUT/f'{a}_latest.pt') for a in parent.ARMS}
    assert initial==terminal['checkpoint_sha256']
    plan=dict(arms=ARMS,seed=SEED,epochs=EPOCHS,batch=BATCH,updates_per_arm=sum(map(len,epochs)),
              original_samples=original,new_samples=48,new_draw_counts=counts,
              replacement='Two positions per batch: old controls retain original samples; mixed arms substitute balanced start examples. Other positions identical.',
              all_original_samples_retained_across_mixed_schedule=True,
              initial_checkpoint_sha256=initial,collection_sha256=digest(collection.RESULT),
              source_sha256={p:digest(p) for p in (__file__,parent.__file__,'scripts/train_go2_dense_task_predictor_development.py')},
              loss='unchanged dense normalized-token L1',encoder_frozen=True,goal_metric_unchanged=True,
              continuation='Restore model and AdamW state from corresponding action or action-blind parent; pair within each parent starts identically.',
              optimizer=dict(name='AdamW',lr=.0003,weight_decay=.01,gradient_clip=1.),
              checkpoint_rule='Fixed final epoch; final model weights only. Optimizer lives in RAM; interrupted runs are failures, not silently restarted.',
              resources=dict(output_free_bytes=free,final_weights_allowance_bytes=350*1024**2,
                             reserve_bytes=512*1024**2,feature_cache='FP16 host RAM only',
                             estimated_wall_minutes=45,cpu_cores=[8,9,10,11],gpu_processes=1),
              evaluation='Compare all four final arms on retained matched-action branches and frozen-goal-metric online control; follow with independent layouts before a navigation claim.',
              limitations=['one seed','coverage intervention, not JEPA encoder objective isolation',
                           'exposed local diagnostics do not establish complete maze navigation'])
    OUTPUT.mkdir();save(OUTPUT/'samples.json',samples);save(OUTPUT/'frame_paths.json',paths);save(OUTPUT/'schedule.json',epochs)
    plan['input_sha256']={n:digest(OUTPUT/n) for n in ('samples.json','frame_paths.json','schedule.json')}
    save(PLAN,plan);save(OUTPUT/'plan.json',plan)
    print('BALANCED_PREDICTOR_PREPARED',plan['updates_per_arm'],counts,flush=True)


def load(arm):
    result=json.loads(RESULT.read_text());assert result['status']=='COMPLETE'
    path=OUTPUT/f'{arm}_final.pt';assert digest(path)==result['checkpoint_sha256'][arm]
    state=torch.load(path,map_location='cpu',weights_only=False);assert state['epoch']==EPOCHS-1
    model=parent.reference.ProprioActionPredictor(use_proprio=False)
    model.load_state_dict(state['model_state_dict'],strict=True)
    return model.eval().requires_grad_(False)


def fit():
    plan=json.loads(PLAN.read_text())
    assert all(digest(p)==h for p,h in plan['source_sha256'].items())
    assert all(digest(OUTPUT/n)==h for n,h in plan['input_sha256'].items())
    assert not (OUTPUT/'process.json').exists()
    save(OUTPUT/'process.json',dict(pid=os.getpid(),plan_sha256=digest(PLAN),affinity=sorted(os.sched_getaffinity(0))))
    torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    started=time.monotonic()
    samples=json.loads((OUTPUT/'samples.json').read_text())
    paths=[Path(p) for p in json.loads((OUTPUT/'frame_paths.json').read_text())]
    epochs=json.loads((OUTPUT/'schedule.json').read_text())
    features,encoding=encode(paths);save(OUTPUT/'encoding.json',encoding)
    indices=torch.tensor([s['frames'] for s in samples]);actions=torch.tensor([s['action'] for s in samples])
    controls=torch.tensor([s['control'] for s in samples])
    models,opts={},{}
    for arm,(_,initial) in ARMS.items():
        path=parent.OUTPUT/f'{initial}_latest.pt';assert digest(path)==plan['initial_checkpoint_sha256'][initial]
        state=torch.load(path,map_location='cpu',weights_only=False)
        model=parent.reference.ProprioActionPredictor(use_proprio=False).cuda().train()
        model.load_state_dict(state['model_state_dict'],strict=True)
        opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);opt.load_state_dict(state['optimizer_state_dict'])
        models[arm],opts[arm]=model,opt;del state
    for initial in parent.ARMS:
        assert all(torch.equal(a,b) for a,b in zip(models[f'old_{initial}'].parameters(),models[f'mixed_{initial}'].parameters(),strict=True))
    histories=[];updates=0
    with (OUTPUT/'progress.jsonl').open('x') as progress:
        for epoch,batches in enumerate(epochs):
            totals={arm:0. for arm in ARMS};count=0
            for batch in batches:
                cpu_rng,cuda_rng=torch.get_rng_state(),torch.cuda.get_rng_state()
                for arm,(data,initial) in ARMS.items():
                    selection=torch.tensor(batch[data]);v=features[indices[selection]].float().cuda()
                    a=actions[selection].cuda();c=controls[selection].cuda()
                    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
                    opt=opts[arm];opt.zero_grad(set_to_none=True)
                    loss=parent.loss_for(models[arm],v[:,:3],a if initial=='action' else torch.zeros_like(a),c,v[:,3])
                    assert torch.isfinite(loss)
                    loss.backward();grad=torch.nn.utils.clip_grad_norm_(models[arm].parameters(),1.);assert torch.isfinite(grad)
                    opt.step();totals[arm]+=float(loss.detach())*len(selection)
                count+=len(batch['old']);updates+=1
                if updates%50==0:print('BALANCED_UPDATES',updates,'epoch',epoch,'elapsed_s',round(time.monotonic()-started,1),flush=True)
            record=dict(epoch=epoch,updates_per_arm=updates,train_l1={a:totals[a]/count for a in ARMS},elapsed_s=time.monotonic()-started)
            histories.append(record);progress.write(json.dumps(record)+'\n');progress.flush()
            print('BALANCED_EPOCH',json.dumps(record),flush=True)
    for arm in ARMS:
        assert shutil.disk_usage(OUTPUT).free> (512+70)*1024**2
        torch.save(dict(model_state_dict=models[arm].state_dict(),epoch=EPOCHS-1,plan_sha256=digest(PLAN)),OUTPUT/f'{arm}_final.pt')
    result=dict(status='COMPLETE',epochs=EPOCHS,updates_per_arm=updates,histories=histories,encoding=encoding,
                checkpoint_sha256={a:digest(OUTPUT/f'{a}_final.pt') for a in ARMS},wall_s=time.monotonic()-started,
                encoder_frozen=True,new_navigation=False,plan_sha256=digest(PLAN))
    save(OUTPUT/'result.json',result);save(RESULT,result)
    print('BALANCED_PREDICTOR_COMPLETE',result['wall_s'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    try:
        prepare() if a.prepare else fit()
    except Exception as error:
        if OUTPUT.exists():
            path=OUTPUT/('prepare_failure.json' if a.prepare else 'fit_failure.json')
            if not path.exists():save(path,dict(reason=repr(error),traceback=traceback.format_exc()))
        raise
