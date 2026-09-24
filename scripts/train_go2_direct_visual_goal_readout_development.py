"""Direct goal-pose readout on the same training pairs and budget as goal cost."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from lewm.direct_visual_goal_readout_development import DirectVisualGoalReadout
from lewm.physical_execution_development import rotation_xyzw
from scripts import train_go2_dense_goal_metric_development as metric

OUTPUT = metric.OUTPUT.parent/'go2_direct_visual_goal_readout_v1_attempt_001'
PLAN = Path('docs/go2_direct_visual_goal_readout_plan_2026-09-17.json')
RESULT = Path('docs/go2_direct_visual_goal_readout_fit_result_2026-09-17.json')
SEED,EPOCHS,BATCH = 2026091709,24,128
save,digest = metric.save,metric.digest


def targets():
    paths,indices,costs,metadata = metric.dataset()
    with np.load(metric.OUTPUT/'pairs.npz',allow_pickle=False) as stored:
        np.testing.assert_array_equal(indices,stored['indices'])
    states = np.zeros((len(paths),3),dtype=np.float64); grouped = defaultdict(list)
    for index,path in enumerate(paths): grouped[path.parent].append((index,int(path.stem.split('_')[1])))
    for directory,entries in grouped.items():
        cameras = json.loads((directory/'camera_audit.json').read_text())
        with np.load(directory/'physics_trace.npz',allow_pickle=False) as archive:
            poses = archive['base_pose_world'].copy()
        for index,frame in entries:
            pose = poses[cameras[frame]['physical_sample_index']]
            rotation = rotation_xyzw(pose[3:])
            states[index] = [pose[0],pose[1],np.arctan2(rotation[1,0],rotation[0,0])]
    def relative(first,second):
        a,b = states[first],states[second]; delta = b-a
        c,s = np.cos(a[:,2]),np.sin(a[:,2])
        return np.stack((c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1],
                         np.arctan2(np.sin(delta[:,2]),np.cos(delta[:,2]))),axis=1)
    forward = relative(indices[:,0],indices[:,1]); reverse = relative(indices[:,1],indices[:,0])
    c,s = np.cos(forward[:,2]),np.sin(forward[:,2])
    composition = forward[:,:2]+np.stack((c*reverse[:,0]-s*reverse[:,1],s*reverse[:,0]+c*reverse[:,1]),axis=1)
    np.testing.assert_allclose(composition,0,rtol=0,atol=1e-12)
    np.testing.assert_allclose(forward[:,2]+reverse[:,2],0,rtol=0,atol=1e-12)
    scale = np.array([.03,.03,np.deg2rad(5)])
    for value in (forward,reverse):
        np.testing.assert_allclose(np.sum((value/scale)**2,axis=1),costs,rtol=1e-6,atol=1e-5)
    return paths,indices,np.stack((forward,reverse),axis=1).astype(np.float32),metadata


def prepare():
    assert not PLAN.exists() and not OUTPUT.exists()
    paths,indices,y,metadata = targets(); torch.manual_seed(SEED)
    model = DirectVisualGoalReadout()
    parameters = sum(p.numel() for p in model.parameters()); assert parameters == 426144
    OUTPUT.mkdir(); np.savez_compressed(OUTPUT/'pairs.npz',indices=indices,targets=y)
    save(OUTPUT/'frame_paths.json',[str(p) for p in paths])
    plan = dict(pairs=len(indices),recordings=len(metadata),frame_paths=len(paths),epochs=EPOCHS,batch=BATCH,seed=SEED,
        parameters=parameters,goal_metric_parameters=426016,extra_parameters=128,
        architecture='shared 1024-to-32 GELU; concatenate current projection and goal-minus-current; 12288-to-32 GELU-to-3; subtract identical-pair output',
        target='signed planar body-frame goal XY and wrapped world-yaw difference; no simulator pose input',
        target_scales=[.03,.03,float(np.deg2rad(5))],loss='MSE in original goal-tolerance units',
        directions='each epoch uses every base pair exactly once; seeded balanced random forward/reverse orientation',
        native_pair_offsets_ms=[100,200,500,1000,2000,3000],encoder_frozen=True,
        optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001,gradient_clip=1.),
        encoder_batch=8,feature_storage='pooled normalized FP16 features in RAM only',
        input='observed current and supplied goal RGB; no action, body, command, predicted feature or oracle pose inputs',
        source_sha256={p:digest(p) for p in (__file__,'lewm/direct_visual_goal_readout_development.py')},
        pairs_sha256=digest(OUTPUT/'pairs.npz'),paths_sha256=digest(OUTPUT/'frame_paths.json'),
        original_pairs_sha256=digest(metric.OUTPUT/'pairs.npz'),
        target_min=y.reshape(-1,3).min(0).tolist(),target_max=y.reshape(-1,3).max(0).tolist(),
        inverse_se2_and_goal_cost_checks_passed=True,train_roles_only=True,
        checkpoint='fixed final epoch; no transfer selection',new_navigation=False,
        execution='prepare labels on CPU while predictor continuation runs; defer GPU fitting until that comparison completes',
        limitations=['same pose-label source as learned cost but signed-vector versus scalar targets',
            'parameter count and updates nearly/exactly matched to cost head, not entire world-model planner',
            'baseline component only; a reactive controller still requires prospective evaluation'])
    save(PLAN,plan); save(OUTPUT/'plan.json',plan)
    print('DIRECT_GOAL_READOUT_PREPARED',json.dumps({k:plan[k] for k in ('pairs','recordings','parameters','target_min','target_max')}),flush=True)


def load():
    terminal = json.loads(RESULT.read_text()); assert terminal['status'] == 'COMPLETE'
    assert digest(OUTPUT/'readout.pt') == terminal['checkpoint_sha256']
    state = torch.load(OUTPUT/'readout.pt',map_location='cpu',weights_only=False); assert state['epoch'] == EPOCHS-1
    model = DirectVisualGoalReadout(); model.load_state_dict(state['model_state_dict'],strict=True)
    return model.eval().requires_grad_(False)


def fit():
    plan = json.loads(PLAN.read_text()); assert all(digest(p)==h for p,h in plan['source_sha256'].items())
    assert digest(OUTPUT/'pairs.npz') == plan['pairs_sha256'] and digest(OUTPUT/'frame_paths.json') == plan['paths_sha256']
    assert not (OUTPUT/'process.json').exists() and not (OUTPUT/'progress.jsonl').exists()
    save(OUTPUT/'process.json',dict(pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),
        output_free_bytes=shutil.disk_usage(OUTPUT).free,gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory))
    torch.set_num_threads(4); torch.manual_seed(SEED); started = time.monotonic()
    paths = [Path(p) for p in json.loads((OUTPUT/'frame_paths.json').read_text())]
    features,encoding = metric.readout.encode(paths,8); save(OUTPUT/'encoding.json',encoding); features = features.cuda()
    with np.load(OUTPUT/'pairs.npz',allow_pickle=False) as a:
        indices = torch.from_numpy(a['indices']).cuda(); targets = torch.from_numpy(a['targets']).cuda()
    model = DirectVisualGoalReadout().cuda().train(); optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    histories = []; updates = 0
    with (OUTPUT/'progress.jsonl').open('x') as stream:
        for epoch in range(EPOCHS):
            start = time.monotonic(); total = 0.
            order = torch.randperm(len(indices),generator=torch.Generator().manual_seed(SEED+epoch))
            flips = (torch.randperm(len(indices),generator=torch.Generator().manual_seed(SEED+1000+epoch)) < len(indices)//2).cuda()
            for chunk in order.split(BATCH):
                chunk = chunk.cuda(); reverse = flips[chunk]
                pair = indices[chunk]; first = torch.where(reverse,pair[:,1],pair[:,0]); second = torch.where(reverse,pair[:,0],pair[:,1])
                target = targets[chunk,reverse.long()]/model.scale
                optimizer.zero_grad(set_to_none=True)
                prediction = model.normalized(features[first].float(),features[second].float())
                loss = F.mse_loss(prediction,target); assert torch.isfinite(loss)
                loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(),1.); assert torch.isfinite(grad)
                optimizer.step(); total += float(loss.detach())*len(chunk); updates += 1
            metric.parent.checkpoint(OUTPUT/'readout.pt',model,optimizer,epoch,plan)
            row = dict(epoch=epoch,updates=updates,normalized_mse=total/len(indices),epoch_s=time.monotonic()-start,
                elapsed_s=time.monotonic()-started)
            histories.append(row); stream.write(json.dumps(row)+'\n'); stream.flush()
            print('DIRECT_GOAL_READOUT_EPOCH',json.dumps(row),flush=True)
    result = dict(status='COMPLETE',epochs=EPOCHS,updates=updates,histories=histories,encoding=encoding,
        wall_s=time.monotonic()-started,checkpoint_sha256=digest(OUTPUT/'readout.pt'),
        no_transfer_data_used=True,encoder_frozen=True,navigation_tested=False)
    save(OUTPUT/'result.json',result); save(RESULT,result)
    print('DIRECT_GOAL_READOUT_COMPLETE',json.dumps({k:v for k,v in result.items() if k!='histories'}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true'); args = parser.parse_args()
    try: prepare() if args.prepare else fit()
    except Exception as error:
        if OUTPUT.exists():
            path = OUTPUT/('prepare_failure.json' if args.prepare else 'fit_failure.json')
            if not path.exists(): save(path,dict(reason=repr(error),traceback=traceback.format_exc()))
        raise
