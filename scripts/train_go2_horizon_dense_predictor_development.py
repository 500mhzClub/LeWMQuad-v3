"""Matched 100--800 ms visual prediction using training-role native recordings."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from lewm.horizon_conditioned_dense_predictor_development import HorizonConditionedDensePredictor
from lewm_genesis.lewm_contract import SafetyLimits, apply_safety_limits_single
from scripts import train_go2_balanced_start_predictor_development as previous
from scripts import collect_go2_balanced_start_horizon_actions_development as collection

parent=previous.parent
OUTPUT=collection.OUTPUT.parent/'go2_horizon_dense_predictor_v1_attempt_001'
PLAN=Path('docs/go2_horizon_dense_predictor_plan_2026-09-18.json')
RESULT=Path('docs/go2_horizon_dense_predictor_result_2026-09-18.json')
SEED,STEPS,BATCH=2026091802,1760,16
ARMS={'action':'mixed_action','no_future_action':'mixed_no_future_action'}
save,digest=previous.save,previous.digest


def dataset():
    groups=defaultdict(list)
    for row in parent.load_training_rows():
        assert row['data_role']=='train' and row['available']
        if row['observation_horizon_receipt']['departure_tick']>=10:
            groups[row['source'],row['trial']].append(row)
    limits=SafetyLimits.from_manifest(yaml.safe_load(Path('config/go2_platform_manifest.yaml').read_text()))
    stats=json.loads((parent.reference.CACHE/'proprio_v1/proprio_norm_stats.json').read_text())
    mean,std=(np.asarray(stats[k],np.float32) for k in ('control_mean','control_std'))
    samples,paths,lookup=[],[],{}

    def append(directory,source,trial,frame,h,commands,measured,available,valid,meta,requested):
        indices=(frame-10,frame-5,frame,frame+h)
        times=[meta['frames'][i]['image_ns'] for i in indices]
        assert np.diff(times[:3]).tolist()==[500_000_000]*2 and times[3]-times[2]==h*100_000_000
        assert measured[frame][[4,9,14]].tolist()==times[:3]
        assert valid[frame].all() and (available[frame]<=times[2]).all()
        assert valid[frame+h,-h:].all()
        applied=np.asarray(apply_safety_limits_single(requested,tuple(commands[frame,-1]),limits)[0],np.float32)
        np.testing.assert_allclose(applied,commands[frame+h,-h:],rtol=0,atol=1e-6)
        assert (applied[:,1]==0).all() and (commands[frame,:,1]==0).all()
        future=np.zeros((8,2),np.float32);future[:h]=applied[:,[0,2]]
        frame_ids=[]
        for i in indices:
            path=str(directory/f'rgb_{i:04d}.png');assert Path(path).is_file()
            if path not in lookup:lookup[path]=len(paths);paths.append(path)
            frame_ids.append(lookup[path])
        samples.append(dict(source=source,trial=trial,frame=frame,horizon=h,frames=frame_ids,
            action=future.tolist(),control=((commands[frame][:,[0,2]].reshape(3,5,2)-mean)/std).tolist()))

    def recording(directory):
        meta=json.loads((directory/'policy_observations.json').read_text())
        with np.load(directory/'policy_histories.npz',allow_pickle=False) as a:
            return (a['applied_command_values'].astype(np.float32),a['applied_command_measured_ns'].copy(),
                    a['applied_command_available_ns'].copy(),a['applied_command_valid'].copy(),meta)

    for (source,trial),rows in sorted(groups.items()):
        directory=parent.ROOTS[source]/trial;arrays=recording(directory)
        for row in sorted(rows,key=lambda r:r['observation_horizon_receipt']['departure_tick']):
            frame=row['observation_horizon_receipt']['departure_tick']
            for h in range(1,9):
                if len(row['known_commands'])<h or len(row['targets'])<h or not row['targets'][h-1]['future_image_valid']:continue
                assert row['targets'][h-1]['future_observation_index']==frame+h
                append(directory,source,trial,frame,h,*arrays,row['known_commands'][:h])
    terminal=json.loads(collection.RESULT.read_text())
    assert terminal['status']=='COMPLETE' and terminal['eligible']==48 and terminal['original_rgb_prefix_matches']
    for case,(trial,action) in enumerate(collection.CASES):
        directory=collection.OUTPUT/f'case_{case:02d}';arrays=recording(directory)
        assert json.loads((directory/'specification.json').read_text())['data_role']=='train'
        for h in range(1,9):
            append(directory,'balanced_start',trial,10,h,*arrays,[collection.candidate_commands(action)[0]]*h)
    counts=Counter((s['source']=='balanced_start',s['horizon']) for s in samples)
    audit=json.loads(Path('docs/go2_native_horizon_training_support_2026-09-18.json').read_text())
    assert all(counts[False,h]==audit['existing_training_offsets'][str(h*100)]['samples'] for h in range(1,9))
    assert all(counts[True,h]==48 for h in range(1,9))
    # Reconstruct the original 500-ms training input exactly, including control order.
    old={ (s['source'],s['trial'],s['frame']):s for s in json.loads((parent.OUTPUT/'samples.json').read_text()) }
    for sample in samples:
        if sample['horizon']==5 and sample['source']!='balanced_start':
            ref=old[sample['source'],sample['trial'],sample['frame']]
            np.testing.assert_allclose(sample['control'],ref['control'],rtol=0,atol=1e-7)
            np.testing.assert_allclose(np.asarray(sample['action'])[:5].reshape(10),ref['action'],rtol=0,atol=1e-7)
    return samples,paths


def schedule(samples):
    rng=np.random.default_rng(SEED);pools=defaultdict(list);queues={};batches=[]
    for i,s in enumerate(samples):pools[s['source']=='balanced_start',s['horizon']].append(i)
    for step in range(STEPS):
        batch=[];start_horizons={step%8+1,(step+4)%8+1}
        for h in range(1,9):
            for draw in range(2):
                key=(h in start_horizons and draw==0,h)
                if not queues.get(key):queues[key]=rng.permutation(pools[key]).tolist()
                batch.append(queues[key].pop())
        rng.shuffle(batch);batches.append(batch)
    assert all(Counter(samples[i]['horizon'] for i in batch)=={h:2 for h in range(1,9)} for batch in batches)
    assert all(sum(samples[i]['source']=='balanced_start' for i in batch)==2 for batch in batches)
    return batches


def compact_save(path,value):
    with path.open('x') as f:json.dump(value,f,separators=(',',':'));f.write('\n')


def prepare():
    assert not OUTPUT.exists() and not PLAN.exists()
    samples,paths=dataset();batches=schedule(samples)
    free=shutil.disk_usage(OUTPUT.parent).free
    estimate=sum((previous.OUTPUT/f'{a}_final.pt').stat().st_size for a in ARMS.values())+40*1024**2
    assert free>512*1024**2+estimate,(free,estimate)
    assert torch.cuda.is_available()
    ram=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    assert ram>len(paths)*768*1024*2+8*1024**3
    OUTPUT.mkdir()
    for name,value in [('samples.json',samples),('frame_paths.json',paths),('schedule.json',batches)]:compact_save(OUTPUT/name,value)
    plan=dict(seed=SEED,updates_per_arm=STEPS,batch=BATCH,arms=ARMS,samples=len(samples),frame_paths=len(paths),
        horizon_ms=list(range(100,801,100)),context_offsets_ms=[-1000,-500,0],
        sampling='Two samples per horizon in every batch; two of sixteen from balanced starts, rotating horizons. Uniform shuffled queues within each pool.',
        draws_by_horizon=dict(Counter(samples[i]['horizon'] for b in batches for i in b)),
        unique_examples_drawn=len({i for b in batches for i in b}),balanced_start_draws=STEPS*2,
        encoder_frozen=True,loss='unchanged L1 of layer-normalized dense visual tokens',
        initialization='Corresponding supplemented action/action-blind final weights; fresh AdamW for both arms. Prior training budgets matched, initial weights differ.',
        initial_checkpoint_sha256={a:digest(previous.OUTPUT/f'{p}_final.pt') for a,p in ARMS.items()},
        optimizer=dict(lr=.0003,weight_decay=.01,gradient_clip=1.),
        action_conditioning='Eight post-limiter forward/yaw commands, hard-masked beyond target time; target-time scalar retained in both arms.',
        source_sha256={p:digest(p) for p in (__file__,'lewm/horizon_conditioned_dense_predictor_development.py',previous.__file__,parent.__file__,'scripts/train_go2_dense_task_predictor_development.py')},
        input_sha256={n:digest(OUTPUT/n) for n in ('samples.json','frame_paths.json','schedule.json')},
        collection_result_sha256=digest(collection.RESULT),
        resources=dict(gpu=torch.cuda.get_device_name(0),free_bytes=free,ram_available_bytes=ram,
            feature_cache_bytes=len(paths)*768*1024*2,reserve_bytes=512*1024**2,gpu_processes=1,cpu_cores=[8,9,10,11]),
        checkpoint_rule='Final step only; model weights retained, no tensor feature files. Preserve interrupted attempts without automatic restart.',
        evaluation='All eight horizons on the retained matched-action branch panel against action-blind and persistence; check 500-ms regression against parent before closed-loop integration.',
        limitations=['single seed','exposed development diagnostics','no JEPA encoder objective ablation',
                     'physical motion decoding, collision scoring and timing remain separate integration problems'])
    save(PLAN,plan);save(OUTPUT/'plan.json',plan)
    print('HORIZON_FIT_PREPARED',len(samples),len(paths),flush=True)


def load(arm):
    result=json.loads(RESULT.read_text());assert result['status']=='COMPLETE'
    path=OUTPUT/f'{arm}_final.pt';assert digest(path)==result['checkpoint_sha256'][arm]
    model=HorizonConditionedDensePredictor(parent.reference.ProprioActionPredictor(use_proprio=False),action_blind=arm=='no_future_action')
    state=torch.load(path,map_location='cpu',weights_only=False);assert state['updates']==STEPS
    model.load_state_dict(state['model_state_dict']);return model.eval().requires_grad_(False)


def fit():
    plan=json.loads(PLAN.read_text())
    assert all(digest(p)==h for p,h in plan['source_sha256'].items())
    assert all(digest(OUTPUT/n)==h for n,h in plan['input_sha256'].items())
    save(OUTPUT/'process.json',dict(pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)),plan_sha256=digest(PLAN)))
    torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
    start=time.monotonic();samples=json.loads((OUTPUT/'samples.json').read_text())
    batches=json.loads((OUTPUT/'schedule.json').read_text())
    features,encoding=previous.encode([Path(p) for p in json.loads((OUTPUT/'frame_paths.json').read_text())])
    save(OUTPUT/'encoding.json',encoding)
    indices=torch.tensor([s['frames'] for s in samples]);actions=torch.tensor([s['action'] for s in samples])
    controls=torch.tensor([s['control'] for s in samples]);horizons=torch.tensor([s['horizon'] for s in samples])
    models={a:HorizonConditionedDensePredictor(previous.load(p),action_blind=a=='no_future_action').cuda().train().requires_grad_(True) for a,p in ARMS.items()}
    opts={a:torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01) for a,m in models.items()}
    totals={a:0. for a in ARMS};history=[]
    with (OUTPUT/'progress.jsonl').open('x') as progress:
        for step,batch in enumerate(batches,1):
            selection=torch.tensor(batch);v=features[indices[selection]].float().cuda()
            action=actions[selection].cuda();control=controls[selection].cuda();h=horizons[selection].cuda()
            mask=torch.ones(BATCH,768,dtype=torch.bool,device='cuda')
            cpu_rng,cuda_rng=torch.get_rng_state(),torch.cuda.get_rng_state()
            for arm,model in models.items():
                torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng);opts[arm].zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(v[:,:3],action,h,mask,control=control)
                loss=F.l1_loss(F.layer_norm(prediction.float(),(1024,)),v[:,3]);assert torch.isfinite(loss)
                loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(grad)
                opts[arm].step();totals[arm]+=float(loss.detach())
            if step%80==0:
                record=dict(updates=step,train_l1={a:totals[a]/80 for a in ARMS},elapsed_s=time.monotonic()-start)
                history.append(record);progress.write(json.dumps(record)+'\n');progress.flush();totals={a:0. for a in ARMS}
                print('HORIZON_UPDATES',json.dumps(record),flush=True)
    for arm,model in models.items():
        assert shutil.disk_usage(OUTPUT).free>(512+70)*1024**2
        torch.save(dict(model_state_dict=model.state_dict(),updates=STEPS,plan_sha256=digest(PLAN)),OUTPUT/f'{arm}_final.pt')
    result=dict(status='COMPLETE',updates_per_arm=STEPS,history=history,encoding=encoding,
        checkpoint_sha256={a:digest(OUTPUT/f'{a}_final.pt') for a in ARMS},wall_s=time.monotonic()-start,
        encoder_frozen=True,new_navigation=False,plan_sha256=digest(PLAN))
    save(OUTPUT/'result.json',result);save(RESULT,result);print('HORIZON_FIT_COMPLETE',result['wall_s'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    try:prepare() if a.prepare else fit()
    except Exception as error:
        if OUTPUT.exists():
            path=OUTPUT/('prepare_failure.json' if a.prepare else 'fit_failure.json')
            if not path.exists():save(path,dict(reason=repr(error),traceback=traceback.format_exc()))
        raise
