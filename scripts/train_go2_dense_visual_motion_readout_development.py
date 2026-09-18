"""Fit one common true-future visual motion probe using existing training data.

Encoder and dynamics checkpoints remain frozen. No candidate actions enter the
probe. Keep pooled features in RAM only; retain a compact final model and scores.
"""
import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from lewm.dense_visual_motion_readout_development import DenseVisualMotionReadout,pool_tokens
from scripts import train_go2_frozen_vjepa_native_adaptation_development as parent

OUTPUT=parent.OUTPUT.parent/'go2_dense_visual_motion_readout_v1_attempt_001'
PLAN=Path('docs/go2_dense_visual_motion_readout_plan_2026-09-17.json')
RESULT=Path('docs/go2_dense_visual_motion_readout_fit_result_2026-09-17.json')
SEED,EPOCHS,BATCH=2026091705,24,64
save,digest=parent.save,parent.digest


def training_data():
    samples=json.loads((parent.OUTPUT/'samples.json').read_text())
    by_id={r['sample_id']:r for r in parent.load_training_rows()}
    targets=[]
    for sample in samples:
        row=by_id[sample['sample_id']];target=row['targets'][4]
        assert row['data_role']=='train' and target['motion_valid'] and target['offset_ns']==500_000_000
        assert target['future_observation_index']==sample['frame']+5
        targets.append(target['motion'])
    assert len(samples)==3518
    return samples,np.asarray(targets,np.float32)


@torch.inference_mode()
def benchmark_encoder(paths):
    encoder=parent.reference.encoders.VJepa21Arm()
    encoder.build(torch.device('cuda:0'),torch.float32)
    pixels=torch.stack([encoder.preprocess(str(p)) for p in paths[:8]]).cuda()
    records=[];reference=None
    for batch in (1,4,8):
        encoder.tokens(pixels[:batch]);torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        values=torch.cat([F.layer_norm(encoder.tokens(pixels[i:i+batch]).float(),(1024,))
                          for i in range(0,len(pixels),batch)])
        torch.cuda.synchronize();seconds=time.monotonic()-started
        if reference is None: reference=values.clone()
        discrepancy=float((values-reference).square().mean())
        assert torch.isfinite(values).all() and discrepancy<1e-8
        records.append(dict(batch=batch,seconds=seconds,frames_per_second=8/seconds,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),mse_vs_batch_one=discrepancy))
        print('READOUT_ENCODER_BENCHMARK',json.dumps(records[-1]),flush=True)
    return records


def prepare():
    assert not PLAN.exists();OUTPUT.mkdir(exist_ok=False)
    terminal=json.loads((parent.OUTPUT/'result.json').read_text())
    assert terminal['status']=='COMPLETE'
    samples,target=training_data()
    paths=[Path(p) for p in json.loads((parent.OUTPUT/'frame_paths.json').read_text())]
    torch.set_num_threads(4)
    benchmark=benchmark_encoder(paths)
    selected=max(benchmark,key=lambda r:r['frames_per_second'])['batch']
    torch.manual_seed(SEED)
    mean,scale=target.mean(0),target.std(0)
    model=DenseVisualMotionReadout(mean,scale)
    plan=dict(schema='dense_visual_motion_readout.v1',samples=len(samples),epochs=EPOCHS,batch=BATCH,
        seed=SEED,encoder_batch=selected,encoder_benchmark=benchmark,
        architecture='2x2 spatial average -> concat current/future-minus-current -> shared 2048x32 GELU -> flatten 192 cells -> 128 GELU -> XY/yaw',
        parameters=sum(p.numel() for p in model.parameters()),
        target_mean=mean.tolist(),target_scale=scale.tolist(),target='body-frame XY metres and yaw radians at 500 ms',
        loss='MSE in training-standardized XY/yaw',optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001,gradient_clip=1.),
        source_sha256={p:digest(p) for p in (__file__,'lewm/dense_visual_motion_readout_development.py')},
        parent_fit_sha256=digest(parent.OUTPUT/'result.json'),
        samples_sha256=digest(parent.OUTPUT/'samples.json'),target_sha256=__import__('hashlib').sha256(target.tobytes()).hexdigest(),
        feature_cache='pooled normalized FP16 features in RAM only',maximum_feature_cache_bytes=len(paths)*192*1024*2,
        neural_inputs='observed current and observed future visual features only; no direct action/body/control inputs',
        planned_evaluation='same head on observed future, adapted action/no-action forecast, persistence; existing command-history and zero-motion controls',
        encoder_and_predictors_frozen=True,fit_uses_predicted_features=False,checkpoint='fixed final epoch',
        no_transfer_selection=True,no_navigation=True,no_new_collection=True,
        limitations=['one seed and readout architecture; decoding failures need not imply information absent',
            '500-ms motion endpoint only, not the eight-horizon runtime contract',
            'training roles only; exposed branch transfer remains development'])
    save(PLAN,plan);save(OUTPUT/'plan.json',plan)
    print('READOUT_PREPARED',len(samples),'samples',plan['parameters'],'parameters',flush=True)


@torch.no_grad()
def encode(paths,batch):
    encoder=parent.reference.encoders.VJepa21Arm()
    encoder.build(torch.device('cuda:0'),torch.float32)
    keys=[digest(p) for p in paths];first={}
    for i,key in enumerate(keys): first.setdefault(key,i)
    unique=list(first.values());features=torch.empty(len(paths),192,1024,dtype=torch.float16)
    started=time.monotonic()
    for start in range(0,len(unique),batch):
        indices=unique[start:start+batch]
        pixels=torch.stack([encoder.preprocess(str(paths[i])) for i in indices]).cuda()
        tokens=F.layer_norm(encoder.tokens(pixels).float(),(1024,))
        assert torch.isfinite(tokens).all()
        features[indices]=pool_tokens(tokens).cpu().half()
        if start//batch%25==0 or start+batch>=len(unique):
            print('READOUT_FEATURES',min(start+batch,len(unique)),len(unique),'seconds',round(time.monotonic()-started,1),flush=True)
    for i,key in enumerate(keys):
        if i!=first[key]: features[i].copy_(features[first[key]])
    del encoder;torch.cuda.empty_cache()
    return features,dict(paths=len(paths),unique=len(unique),seconds=time.monotonic()-started)


def load():
    record=json.loads((OUTPUT/'result.json').read_text());plan=json.loads(PLAN.read_text())
    assert record['status']=='COMPLETE' and digest(OUTPUT/'readout.pt')==record['model_sha256']
    model=DenseVisualMotionReadout(plan['target_mean'],plan['target_scale'])
    state=torch.load(OUTPUT/'readout.pt',map_location='cpu',weights_only=False)
    assert state['epoch']==EPOCHS-1
    model.load_state_dict(state['model_state_dict'],strict=True)
    return model.eval().requires_grad_(False)


def fit():
    plan=json.loads(PLAN.read_text())
    for path,sha in plan['source_sha256'].items(): assert digest(path)==sha
    assert not (OUTPUT/'progress.jsonl').exists()
    samples,target=training_data()
    assert __import__('hashlib').sha256(target.tobytes()).hexdigest()==plan['target_sha256']
    assert digest(parent.OUTPUT/'samples.json')==plan['samples_sha256']
    torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
    started=time.monotonic()
    paths=[Path(p) for p in json.loads((parent.OUTPUT/'frame_paths.json').read_text())]
    features,encoding=encode(paths,plan['encoder_batch'])
    save(OUTPUT/'encoding.json',encoding)
    features=features.cuda()
    indices=torch.tensor([[s['frames'][2],s['frames'][3]] for s in samples],device='cuda')
    y=torch.from_numpy(target).cuda()
    model=DenseVisualMotionReadout(plan['target_mean'],plan['target_scale']).cuda().train()
    normalized=(y-model.target_mean)/model.target_scale
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    histories=[];updates=0
    with (OUTPUT/'progress.jsonl').open('x') as progress:
        for epoch in range(EPOCHS):
            epoch_start=time.monotonic();total=0.
            order=torch.randperm(len(samples),generator=torch.Generator().manual_seed(SEED+epoch))
            for chunk in order.split(BATCH):
                chunk=chunk.cuda();pairs=features[indices[chunk]].float()
                optimizer.zero_grad(set_to_none=True)
                loss=F.mse_loss(model.normalized(pairs[:,0],pairs[:,1]),normalized[chunk])
                assert torch.isfinite(loss)
                loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                assert torch.isfinite(grad)
                optimizer.step();total+=float(loss.detach())*len(chunk);updates+=1
            histories.append(total/len(samples))
            temporary=OUTPUT/'readout.tmp'
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                epoch=epoch,plan=plan),temporary);temporary.replace(OUTPUT/'readout.pt')
            row=dict(epoch=epoch,train_normalized_mse=histories[-1],updates=updates,
                epoch_s=time.monotonic()-epoch_start,elapsed_s=time.monotonic()-started)
            progress.write(json.dumps(row)+'\n');progress.flush()
            print('READOUT_EPOCH',json.dumps(row),flush=True)
    result=dict(status='COMPLETE',epochs=EPOCHS,updates=updates,training_samples=len(samples),
        model_sha256=digest(OUTPUT/'readout.pt'),encoding=encoding,histories=histories,
        wall_s=time.monotonic()-started,encoder_and_predictors_unchanged=True,
        transfer_evaluated=False,navigation_tested=False)
    save(OUTPUT/'result.json',result);save(RESULT,result)
    print('READOUT_FIT_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    try: prepare() if args.prepare else fit()
    except Exception as error:
        if OUTPUT.exists() and not (OUTPUT/'failure.json').exists():
            save(OUTPUT/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc()))
        raise
