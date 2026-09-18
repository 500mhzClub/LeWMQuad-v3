"""Matched old/mixed-data continuation of the original frozen-feature head."""
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from scripts import collect_go2_full_heading_training_development as collection
from scripts import train_go2_dense_visual_motion_readout_development as original

OUTPUT = collection.OUTPUT.parent/'go2_full_heading_readout_v1_attempt_001'
STEPS, BATCH, SEED = 440, 64, 2026091806
ARMS = ('old_data','mixed_data')


def cycle(n,count,seed):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(n) for _ in range((count+n-1)//n)])[:count]


def dataset():
    samples,targets = original.training_data()
    old_paths = json.loads((original.parent.OUTPUT/'frame_paths.json').read_text())
    new = json.loads((collection.OUTPUT/'samples.json').read_text())
    assert new and all(r['data_role']=='train' and r['horizon_ms']==500 for r in new)
    paths = []; lookup = {}; pairs = []
    def add(path):
        path = str(Path(path).resolve())
        if path not in lookup:
            lookup[path] = len(paths);paths.append(path)
        return lookup[path]
    for row in samples:
        pairs.append([add(old_paths[row['frames'][i]]) for i in (2,3)])
    for row in new:
        pairs.append([add(row[k]) for k in ('current_rgb','future_rgb')])
    y = np.concatenate((targets,np.asarray([r['motion'] for r in new],np.float32)))
    assert np.isfinite(y).all()
    return paths,torch.tensor(pairs,dtype=torch.long),torch.from_numpy(y),len(samples),len(new)


def load(arm):
    if arm not in ARMS:
        raise ValueError('explicit continuation arm required')
    result = json.loads((OUTPUT/'result.json').read_text())
    path = OUTPUT/f'{arm}_final.pt'
    assert result['status']=='COMPLETE' and original.digest(path)==result['checkpoint_sha256'][arm]
    model = original.load()
    state = torch.load(path,map_location='cpu',weights_only=False)
    assert state['updates']==STEPS
    model.load_state_dict(state['model_state_dict'])
    return model.eval().requires_grad_(False)


def main():
    terminal = json.loads((collection.OUTPUT/'result.json').read_text())
    assert terminal['status']=='COMPLETE'
    paths,pairs,targets,old_count,new_count = dataset()
    assert original.digest(collection.OUTPUT/'samples.json')==terminal['samples_sha256']
    cache_bytes = len(paths)*192*1024*2
    assert psutil.virtual_memory().available > cache_bytes+8*1024**3
    assert shutil.disk_usage(OUTPUT.parent).free > 512*1024**2
    OUTPUT.mkdir(exist_ok=False)
    common = cycle(old_count,STEPS*32,SEED).reshape(STEPS,32)
    extra = cycle(old_count,STEPS*32,SEED+1).reshape(STEPS,32)
    new = cycle(new_count,STEPS*32,SEED+2).reshape(STEPS,32)+old_count
    schedules = dict(old_data=np.concatenate((common,extra),axis=1),mixed_data=np.concatenate((common,new),axis=1))
    assert len(np.unique(common))==old_count
    assert len(np.unique(new))==new_count
    encoder_batch = json.loads(original.PLAN.read_text())['encoder_batch']
    plan = dict(arms=ARMS,updates_per_arm=STEPS,batch=BATCH,seed=SEED,
        old_samples=old_count,new_samples=new_count,feature_paths=len(paths),feature_cache_bytes=cache_bytes,
        shared_old_examples_per_batch=32,mixed_new_examples_per_batch=32,
        every_old_example_seen_in_both_arms=True,every_new_example_seen_in_mixed_arm=True,
        initial_head_sha256=original.digest(original.OUTPUT/'readout.pt'),
        training_collection_result_sha256=original.digest(collection.OUTPUT/'result.json'),
        new_samples_sha256=terminal['samples_sha256'],source_sha256=original.digest(__file__),
        old_target_normalization_retained=True,encoder_and_predictor_unchanged=True,
        optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001,gradient_clip=1.),
        loss='MSE in original training-standardized XY/yaw',encoder_batch=encoder_batch,
        feature_cache='pooled FP16 features in RAM only',checkpoint='fixed final step, no selection on development scores',
        comparison='extra optimization controlled; mixed arm combines wider headings with visible-robot rendering',
        evaluation='preserved exposed branch/pilot diagnostics before any new closed-loop attempt',
        no_eval_maze_training=True,independent_geometry_or_JEPA_objective_claim=False)
    original.save(OUTPUT/'plan.json',plan)
    original.save(OUTPUT/'process.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),affinity=psutil.Process().cpu_affinity()))
    torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
    started = time.monotonic()
    try:
        features,encoding = original.encode([Path(p) for p in paths],encoder_batch)
        original.save(OUTPUT/'encoding.json',encoding)
        models = {a:original.load().cuda().train().requires_grad_(True) for a in ARMS}
        for key,value in models['old_data'].state_dict().items():
            assert torch.equal(value,models['mixed_data'].state_dict()[key])
        optimizers = {a:torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001) for a,m in models.items()}
        totals = {a:0. for a in ARMS};history = []
        with (OUTPUT/'progress.jsonl').open('x') as progress:
            for step in range(STEPS):
                for arm in ARMS:
                    indices = torch.from_numpy(schedules[arm][step])
                    x = features[pairs[indices]].float().cuda();model = models[arm]
                    y = (targets[indices].cuda()-model.target_mean)/model.target_scale
                    optimizer = optimizers[arm];optimizer.zero_grad(set_to_none=True)
                    loss = F.mse_loss(model.normalized(x[:,0],x[:,1]),y)
                    assert torch.isfinite(loss)
                    loss.backward();norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    assert torch.isfinite(norm)
                    optimizer.step();totals[arm] += float(loss.detach())
                if (step+1)%55==0:
                    row = dict(updates=step+1,train_normalized_mse={a:v/55 for a,v in totals.items()},wall_s=time.monotonic()-started)
                    history.append(row);progress.write(json.dumps(row)+'\n');progress.flush()
                    totals = {a:0. for a in ARMS};print('FULL_HEADING_READOUT_UPDATES',json.dumps(row),flush=True)
        for arm,model in models.items():
            torch.save(dict(model_state_dict=model.cpu().state_dict(),updates=STEPS,plan_sha256=original.digest(OUTPUT/'plan.json')),OUTPUT/f'{arm}_final.pt')
        result = dict(status='COMPLETE',updates_per_arm=STEPS,history=history,encoding=encoding,
            checkpoint_sha256={a:original.digest(OUTPUT/f'{a}_final.pt') for a in ARMS},
            wall_s=time.monotonic()-started,encoder_and_predictor_unchanged=True,
            no_new_navigation=True,development_evaluation_pending=True)
        original.save(OUTPUT/'result.json',result)
        print('FULL_HEADING_READOUT_COMPLETE',result['wall_s'],flush=True)
    except BaseException as error:
        original.save(OUTPUT/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
