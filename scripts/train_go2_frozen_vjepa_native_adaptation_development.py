"""Matched native-domain adaptation of existing dense V-JEPA dynamics.

Frozen encoder; same initialization/order/budget for action and action-blind
predictors. Existing training-role RGB only. Features live in RAM, and only the
latest resumable checkpoint per arm is retained. No simulation or navigation.
"""
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from lewm_genesis.lewm_contract import SafetyLimits, apply_safety_limits_single
from scripts import evaluate_go2_frozen_vjepa_native_branches_development as reference
from scripts.prepare_go2_short_pulse_training_development import ROOTS, load_training_rows

OUTPUT = Path('/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_frozen_vjepa_native_adaptation_v1_attempt_001')
PLAN = Path('docs/go2_frozen_vjepa_native_adaptation_plan_2026-09-17.json')
RESULT = Path('docs/go2_frozen_vjepa_native_adaptation_result_2026-09-17.json')
INITIAL = reference.CACHE/f'factorial_v1/seed_{reference.SEED}/seed_{reference.SEED}_rgb_rollout_epoch21.pt'
ARMS = ('action', 'no_future_action')
EPOCHS, BATCH, SEED = 24, 16, 2026091704
save, digest = reference.save, reference.digest


def dataset():
    """Construct samples from previously admitted train-only observation windows."""
    groups = defaultdict(list)
    for row in load_training_rows():
        f = row['observation_horizon_receipt']['departure_tick']
        if f >= 10 and len(row['known_commands']) >= 5 and row['targets'][4]['future_image_valid']:
            assert row['available'] and row['data_role'] == 'train'
            groups[row['source'], row['trial']].append(row)
    limits = SafetyLimits.from_manifest(yaml.safe_load(Path('config/go2_platform_manifest.yaml').read_text()))
    stats = json.loads((reference.CACHE/'proprio_v1/proprio_norm_stats.json').read_text())
    mean, std = (np.asarray(stats[k], np.float32) for k in ('control_mean', 'control_std'))
    samples, paths, lookup = [], [], {}
    for (source, trial), rows in sorted(groups.items()):
        directory = ROOTS[source]/trial
        meta = json.loads((directory/'policy_observations.json').read_text())
        with np.load(directory/'policy_histories.npz', allow_pickle=False) as a:
            commands = a['applied_command_values'].astype(np.float32)
            times = a['applied_command_measured_ns'].copy()
            available = a['applied_command_available_ns'].copy()
            valid = a['applied_command_valid'].copy()
        for row in sorted(rows, key=lambda r:r['observation_horizon_receipt']['departure_tick']):
            f = row['observation_horizon_receipt']['departure_tick']
            frames = [f-10, f-5, f, f+5]
            assert row['targets'][4]['future_observation_index'] == f+5
            image_times = [meta['frames'][i]['image_ns'] for i in frames]
            assert np.diff(image_times).tolist() == [500_000_000]*3
            assert times[f][[4,9,14]].tolist() == image_times[:3]
            assert valid[f].all() and (available[f] <= image_times[2]).all()
            requested = row['known_commands'][:5]
            applied, _ = apply_safety_limits_single(requested, tuple(commands[f,-1]), limits)
            applied = np.asarray(applied, np.float32)
            np.testing.assert_allclose(applied, commands[f+5,-5:], atol=1e-6, rtol=0)
            assert (applied[:,1] == 0).all() and (commands[f,:,1] == 0).all()
            indices = []
            for frame in frames:
                path = directory/f'rgb_{frame:04d}.png'
                if path not in lookup:
                    lookup[path] = len(paths)
                    paths.append(path)
                indices.append(lookup[path])
            control = (commands[f][:,[0,2]].reshape(3,5,2)-mean)/std
            samples.append(dict(sample_id=row['sample_id'],source=source,trial=trial,
                                frame=f,frames=indices,action=applied[:,[0,2]].reshape(10).tolist(),
                                control=control.tolist()))
    assert len({(r['source'],r['trial'],r['frame']) for r in samples}) == len(samples)
    return samples, paths


def model():
    result = reference.ProprioActionPredictor(use_proprio=False)
    state = torch.load(INITIAL, map_location='cpu', weights_only=False)
    result.load_state_dict(state['model_state_dict'], strict=True)
    return result


def loss_for(m, context, action, control, target):
    mask = torch.ones(len(context),768,dtype=torch.bool,device=context.device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        prediction = m(context, action, mask, control=control)
        prediction = F.layer_norm(prediction.float(), (1024,))
        return F.l1_loss(prediction, target)


def benchmark():
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    result = []
    for micro in (4,8,16):
        m = model().cuda().train()
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=.01)
        x = torch.randn(micro,3,768,1024,device='cuda')
        x = F.layer_norm(x,(1024,))
        y = torch.randn(micro,768,1024,device='cuda')
        y = F.layer_norm(y,(1024,))
        a = torch.zeros(micro,10,device='cuda')
        c = torch.zeros(micro,3,5,2,device='cuda')
        torch.cuda.reset_peak_memory_stats()
        durations = []
        try:
            for step in range(4):
                torch.cuda.synchronize();start = time.monotonic()
                opt.zero_grad(set_to_none=True)
                loss = loss_for(m,x,a,c,y)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
                assert torch.isfinite(loss) and torch.isfinite(grad)
                opt.step();torch.cuda.synchronize()
                if step:durations.append(time.monotonic()-start)
            record = dict(microbatch=micro,step_seconds=float(np.median(durations)),
                          samples_per_second=micro/float(np.median(durations)),
                          peak_gpu_bytes=torch.cuda.max_memory_allocated(),status='PASS')
        except torch.cuda.OutOfMemoryError:
            record = dict(microbatch=micro,status='OUT_OF_MEMORY')
        result.append(record)
        print('MICROBATCH_BENCHMARK',json.dumps(record),flush=True)
        del m,opt,x,y,a,c
        torch.cuda.empty_cache()
    return result


def prepare():
    OUTPUT.mkdir(exist_ok=False)
    samples, paths = dataset()
    save(OUTPUT/'samples.json',samples)
    save(OUTPUT/'frame_paths.json',[str(p) for p in paths])
    speed = benchmark()
    passed = [r for r in speed if r['status']=='PASS']
    chosen = max(passed,key=lambda r:r['samples_per_second'])['microbatch']
    plan = dict(schema='frozen_vjepa_native_adaptation.v1',arms=ARMS,epochs=EPOCHS,
        effective_batch=BATCH,microbatch=chosen,seed=SEED,samples=len(samples),
        recordings=len({(r['source'],r['trial']) for r in samples}),frame_paths=len(paths),
        max_feature_cache_bytes=len(paths)*768*1024*2,feature_storage='RAM only, normalized FP16',
        encoder_precision='float32',predictor_precision='bf16 autocast with float32 loss',
        objective='one-step dense L1 of layer-normalized visual tokens',
        native_horizon_ms=500,context_offsets_ms=[-1000,-500,0],
        optimizer=dict(name='AdamW',lr=3e-4,weight_decay=.01,gradient_clip=1.),
        initialization=str(INITIAL),initialization_sha256=digest(INITIAL),
        source_sha256=digest(__file__),samples_sha256=digest(OUTPUT/'samples.json'),
        control_normalization='unchanged historical training-only statistics',
        input_difference='no_future_action sets the future command block to zero; past control and RGB unchanged',
        checkpoint_rule='fixed final epoch; no selection using native transfer outcomes',
        resources=dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
                       gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory),
        parallelism='one GPU process, paired arms interleaved on identical batches; four CPU threads',
        benchmark=speed,no_simulation=True,encoder_frozen=True,new_collection=False,
        limitations=['native training-role data only; exposed transfer panel remains development',
                     'one preselected initialization/seed',
                     'predictor adaptation does not establish JEPA representation-learning benefit',
                     'single-step native adaptation of a historically rollout-trained initialization'])
    save(OUTPUT/'plan.json',plan);save(PLAN,plan)
    print('NATIVE_ADAPTATION_PREPARED',len(samples),'sequences',len(paths),'frame paths',flush=True)


@torch.no_grad()
def encode_frames(paths):
    encoder = reference.encoders.VJepa21Arm()
    encoder.build(torch.device('cuda:0'),torch.float32)
    cache = torch.empty(len(paths),768,1024,dtype=torch.float16)
    # Reuse identical RGB exactly; frame paths still retain their episode identity.
    seen = {};started=time.monotonic()
    for i,path in enumerate(paths):
        key = digest(path)
        if key in seen:
            cache[i].copy_(cache[seen[key]])
        else:
            pixel = encoder.preprocess(str(path))[None].cuda()
            tokens = encoder.tokens(pixel).float()
            assert tuple(tokens.shape) == (1,768,1024) and torch.isfinite(tokens).all()
            cache[i].copy_(F.layer_norm(tokens,(1024,))[0].cpu().half())
            seen[key]=i
        if (i+1)%200==0 or i+1==len(paths):
            print('NATIVE_FEATURES',i+1,len(paths),'unique',len(seen),'seconds',round(time.monotonic()-started,1),flush=True)
    del encoder
    torch.cuda.empty_cache()
    return cache,dict(frame_paths=len(paths),unique_frames=len(seen),seconds=time.monotonic()-started)


def checkpoint(path, m, opt, epoch, plan):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(model_state_dict=m.state_dict(),optimizer_state_dict=opt.state_dict(),
        epoch=epoch,plan=plan,torch_rng_state=torch.get_rng_state(),
        cuda_rng_state=torch.cuda.get_rng_state_all(),numpy_rng_state=np.random.get_state(),
        python_rng_state=random.getstate()),temporary)
    temporary.replace(path)


def fit():
    plan=json.loads(PLAN.read_text())
    assert digest(__file__)==plan['source_sha256']
    assert digest(INITIAL)==plan['initialization_sha256']
    assert digest(OUTPUT/'samples.json')==plan['samples_sha256']
    if (OUTPUT/'progress.jsonl').exists():
        raise ValueError('existing training attempt; inspect state rather than restart')
    samples=json.loads((OUTPUT/'samples.json').read_text())
    paths=[Path(p) for p in json.loads((OUTPUT/'frame_paths.json').read_text())]
    torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    started=time.monotonic()
    features,encoding=encode_frames(paths)
    save(OUTPUT/'encoding.json',encoding)
    frame_indices=torch.tensor([r['frames'] for r in samples])
    action=torch.tensor([r['action'] for r in samples],dtype=torch.float32)
    control=torch.tensor([r['control'] for r in samples],dtype=torch.float32)
    base=model()
    models={a:copy.deepcopy(base).cuda().train() for a in ARMS};del base
    optimizers={a:torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.01) for a,m in models.items()}
    histories={a:[] for a in ARMS};updates=0
    with (OUTPUT/'progress.jsonl').open('x') as progress:
        for epoch in range(EPOCHS):
            epoch_start=time.monotonic()
            order=torch.randperm(len(samples),generator=torch.Generator().manual_seed(SEED+epoch))
            totals={a:0. for a in ARMS}
            for offset in range(0,len(order),BATCH):
                selection=order[offset:offset+BATCH]
                for arm in ARMS:optimizers[arm].zero_grad(set_to_none=True)
                for chunk in selection.split(plan['microbatch']):
                    v=features[frame_indices[chunk]].float().cuda()
                    x,y=v[:,:3],v[:,3]
                    a=action[chunk].cuda();c=control[chunk].cuda()
                    for arm in ARMS:
                        commands=a if arm=='action' else torch.zeros_like(a)
                        loss=loss_for(models[arm],x,commands,c,y)
                        if not torch.isfinite(loss):raise ValueError(f'nonfinite {arm} loss')
                        (loss*len(chunk)/len(selection)).backward()
                        totals[arm]+=float(loss.detach())*len(chunk)
                for arm in ARMS:
                    grad=torch.nn.utils.clip_grad_norm_(models[arm].parameters(),1.)
                    if not torch.isfinite(grad):raise ValueError(f'nonfinite {arm} gradients')
                    optimizers[arm].step()
                updates+=1
                if updates%50==0:
                    print('NATIVE_ADAPTATION_UPDATES',updates,'epoch',epoch,'elapsed_s',round(time.monotonic()-started,1),flush=True)
            for arm in ARMS:
                histories[arm].append(totals[arm]/len(samples))
                checkpoint(OUTPUT/f'{arm}_latest.pt',models[arm],optimizers[arm],epoch,plan)
            record=dict(epoch=epoch,updates_per_arm=updates,
                        train_l1={a:histories[a][-1] for a in ARMS},
                        epoch_s=time.monotonic()-epoch_start,elapsed_s=time.monotonic()-started)
            progress.write(json.dumps(record)+'\n');progress.flush()
            print('NATIVE_ADAPTATION_EPOCH',json.dumps(record),flush=True)
    result=dict(status='COMPLETE',epochs=EPOCHS,updates_per_arm=updates,
                sequences=len(samples),histories=histories,encoding=encoding,
                wall_s=time.monotonic()-started,encoder_frozen=True,
                checkpoint_sha256={a:digest(OUTPUT/f'{a}_latest.pt') for a in ARMS},
                no_transfer_data_used=True,navigation_tested=False)
    save(OUTPUT/'result.json',result);save(RESULT,result)
    print('NATIVE_ADAPTATION_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    try:
        prepare() if args.prepare else fit()
    except Exception as error:
        if OUTPUT.exists():
            path=OUTPUT/('prepare_failure.json' if args.prepare else 'fit_failure.json')
            if not path.exists():save(path,dict(reason=repr(error),traceback=traceback.format_exc()))
        raise
