"""Fit a physical goal metric on fixed training-layout image pairs only."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from lewm.dense_goal_metric_development import DenseGoalMetric
from lewm.physical_execution_development import rotation_xyzw
from scripts import train_go2_dense_visual_motion_readout_development as readout

parent = readout.parent
save, digest = parent.save, parent.digest
OUTPUT = parent.OUTPUT.parent/'go2_dense_goal_metric_v1_attempt_001'
PLAN = Path('docs/go2_dense_goal_metric_plan_2026-09-17.json')
RESULT = Path('docs/go2_dense_goal_metric_fit_result_2026-09-17.json')
SEED, EPOCHS, BATCH = 2026091707, 24, 128
OFFSETS = (1, 2, 5, 10, 20, 30)
POSITION_SCALE, HEADING_SCALE = .03, np.deg2rad(5.)


def dataset():
    samples = json.loads((parent.OUTPUT/'samples.json').read_text())
    admitted = {r['sample_id']:r for r in parent.load_training_rows()}
    directories = {}
    for s in samples:
        row = admitted[s['sample_id']]
        assert row['data_role'] == 'train'
        directory = parent.ROOTS[s['source']]/s['trial']
        directories[directory] = dict(source=s['source'], trial=s['trial'], role='train')
    paths = [Path(p) for p in json.loads((parent.OUTPUT/'frame_paths.json').read_text())]
    grouped = defaultdict(dict)
    for index, path in enumerate(paths):
        assert path.parent in directories and path.name.startswith('rgb_') and path.suffix == '.png'
        grouped[path.parent][int(path.stem.split('_')[1])] = index
    pairs, targets, metadata = [], [], []
    for directory, frames in sorted(grouped.items()):
        camera = json.loads((directory/'camera_audit.json').read_text())
        with np.load(directory/'physics_trace.npz', allow_pickle=False) as archive:
            poses = archive['base_pose_world']; contacts = archive['physics_contact']; times = archive['timestamp_s']
        contact_seen = np.maximum.accumulate(contacts)
        states = {}
        for frame in frames:
            at = camera[frame]['physical_sample_index']
            if contact_seen[at]:
                continue
            assert abs(times[at]-camera[frame]['timestamp_s']) < 1e-9
            pose = poses[at]
            rotation = rotation_xyzw(pose[3:])
            states[frame] = np.array([pose[0], pose[1], np.arctan2(rotation[1,0],rotation[0,0])])
        counts = Counter()
        for first in sorted(states):
            for offset in OFFSETS:
                second = first+offset
                if second not in states:
                    continue
                delta = states[second]-states[first]
                yaw = np.arctan2(np.sin(delta[2]),np.cos(delta[2]))
                cost = np.sum(delta[:2]**2)/POSITION_SCALE**2 + yaw**2/HEADING_SCALE**2
                pairs.append([frames[first],frames[second]])
                targets.append(float(cost)); counts[offset] += 1
        metadata.append(directories[directory] | dict(pairs=sum(counts.values()),offset_counts=dict(counts)))
    indices, target = np.asarray(pairs,np.int64), np.asarray(targets,np.float32)
    assert len(indices) and np.isfinite(target).all() and (target >= 0).all()
    return paths, indices, target, metadata


def prepare():
    assert not PLAN.exists() and not OUTPUT.exists()
    paths, indices, target, metadata = dataset()
    torch.manual_seed(SEED)
    model = DenseGoalMetric()
    benchmark = json.loads(readout.PLAN.read_text())
    assert benchmark['encoder_batch'] == 8
    OUTPUT.mkdir()
    np.savez_compressed(OUTPUT/'pairs.npz',indices=indices,target=target)
    save(OUTPUT/'frame_paths.json',[str(p) for p in paths]); save(OUTPUT/'recordings.json',metadata)
    plan = dict(pairs=len(indices),recordings=len(metadata),frame_paths=len(paths),epochs=EPOCHS,batch=BATCH,seed=SEED,
        offsets_ms=[100*i for i in OFFSETS],encoder_batch=8,
        architecture='2x2 mean pooled normalized tokens; shared 1024-to-32 GELU, flatten 192 cells, linear 64-dimensional embedding; mean squared embedding distance',
        parameters=sum(p.numel() for p in model.parameters()),
        target='squared world-planar XY separation / .03^2 plus squared wrapped world-yaw separation / radians(5)^2',
        position_scale_m=POSITION_SCALE,heading_scale_rad=float(HEADING_SCALE),
        scales_from_existing_pilot_tolerances=True,
        loss='MSE between log1p predicted squared metric distance and log1p physical target distance',
        optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001,gradient_clip=1.),
        source_sha256={p:digest(p) for p in (__file__,'lewm/dense_goal_metric_development.py',
            'scripts/train_go2_dense_visual_motion_readout_development.py','lewm/dense_visual_motion_readout_development.py')},
        parent_training_samples_sha256=digest(parent.OUTPUT/'samples.json'),pairs_sha256=digest(OUTPUT/'pairs.npz'),
        paths_sha256=digest(OUTPUT/'frame_paths.json'),
        target_quantiles=np.quantile(target,[0,.25,.5,.75,.9,1]).tolist(),
        source_pair_counts=dict(Counter({s:sum(m['pairs'] for m in metadata if m['source']==s) for s in {m['source'] for m in metadata}})),
        encoder_and_predictors_frozen=True,fit_inputs='two observed image representations only; no action/body/control/pose inputs',
        labels='simulator poses used only as training targets; pair endpoints must precede any contact',
        checkpoint='fixed final epoch; no transfer selection',
        features='FP16 pooled features in RAM only; float32 goal metric training',
        maximum_feature_cache_bytes=len(paths)*192*1024*2,
        resources=dict(available_ram_gib=72,output_free_gib=2.6,root_free_gib=1.2,gpu_vram_gib=31.86,competing_compute_jobs=0),
        concurrency='one GPU process, four CPU threads on cores 8-11; reuse proven batch-eight extraction (about five distinct images per second)',
        evaluation='same frozen cost on actual and action/no-action predicted future features; retain raw dense-MSE control',
        additional_supervision_is_not_jepa_objective=True,new_navigation=False,
        limitations=['one architecture and seed; small training geometry family',
            'a learned symmetric distance is not a stopping detector or collision predictor',
            'pair distance supervision does not establish JEPA-training superiority'])
    save(PLAN,plan); save(OUTPUT/'plan.json',plan)
    print('GOAL_METRIC_PREPARED',json.dumps({k:plan[k] for k in ('pairs','recordings','frame_paths','parameters','target_quantiles','source_pair_counts')}),flush=True)


def load():
    result = json.loads((OUTPUT/'result.json').read_text())
    assert result['status'] == 'COMPLETE' and digest(OUTPUT/'metric.pt') == result['checkpoint_sha256']
    state = torch.load(OUTPUT/'metric.pt',map_location='cpu',weights_only=False)
    assert state['epoch'] == EPOCHS-1
    model = DenseGoalMetric(); model.load_state_dict(state['model_state_dict'],strict=True)
    return model.eval().requires_grad_(False)


def fit():
    plan = json.loads(PLAN.read_text())
    assert all(digest(p)==h for p,h in plan['source_sha256'].items())
    assert digest(OUTPUT/'pairs.npz') == plan['pairs_sha256'] and digest(OUTPUT/'frame_paths.json') == plan['paths_sha256']
    assert not (OUTPUT/'progress.jsonl').exists()
    torch.set_num_threads(4); torch.manual_seed(SEED)
    started = time.monotonic()
    paths = [Path(p) for p in json.loads((OUTPUT/'frame_paths.json').read_text())]
    features, encoding = readout.encode(paths,8)
    save(OUTPUT/'encoding.json',encoding)
    features = features.cuda()
    with np.load(OUTPUT/'pairs.npz',allow_pickle=False) as archive:
        indices = torch.from_numpy(archive['indices']).cuda()
        target = torch.from_numpy(archive['target']).cuda()
    model = DenseGoalMetric().cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    histories=[]; updates=0
    with (OUTPUT/'progress.jsonl').open('x') as progress:
        for epoch in range(EPOCHS):
            epoch_started=time.monotonic();total=0.
            order=torch.randperm(len(indices),generator=torch.Generator().manual_seed(SEED+epoch))
            for chunk in order.split(BATCH):
                chunk=chunk.cuda();first,second=features[indices[chunk]].float().unbind(1)
                optimizer.zero_grad(set_to_none=True)
                predicted=model(first,second)
                loss=F.mse_loss(torch.log1p(predicted),torch.log1p(target[chunk]))
                assert torch.isfinite(loss)
                loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                assert torch.isfinite(grad)
                optimizer.step();total+=float(loss.detach())*len(chunk);updates+=1
            histories.append(total/len(indices))
            temporary=OUTPUT/'metric.tmp'
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                epoch=epoch,plan=plan,rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state()),temporary)
            temporary.replace(OUTPUT/'metric.pt')
            row=dict(epoch=epoch,train_log_distance_mse=histories[-1],updates=updates,
                epoch_s=time.monotonic()-epoch_started,elapsed_s=time.monotonic()-started)
            progress.write(json.dumps(row)+'\n');progress.flush();print('GOAL_METRIC_EPOCH',json.dumps(row),flush=True)
    model.eval()
    with torch.inference_mode():
        sample=features[:8].float()
        assert torch.equal(model(sample,sample),torch.zeros(8,device='cuda'))
        assert torch.equal(model(sample,sample.flip(0)),model(sample.flip(0),sample))
    result=dict(status='COMPLETE',epochs=EPOCHS,updates=updates,pairs=len(indices),histories=histories,
        encoding=encoding,wall_s=time.monotonic()-started,checkpoint_sha256=digest(OUTPUT/'metric.pt'),
        encoder_and_predictors_unchanged=True,transfer_evaluated=False,navigation_tested=False,
        zero_identity_and_symmetry_verified=True)
    save(OUTPUT/'result.json',result);save(RESULT,result)
    print('GOAL_METRIC_FIT_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    try: prepare() if args.prepare else fit()
    except Exception as error:
        if OUTPUT.exists() and not (OUTPUT/'failure.json').exists():
            save(OUTPUT/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc()))
        raise
