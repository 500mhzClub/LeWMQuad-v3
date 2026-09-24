"""Matched temporal readout coverage on existing original and heading train data."""
import argparse
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

from scripts import train_go2_multihorizon_motion_readout_development as prior

PREDECESSOR = prior.OUTPUT
OUTPUT = PREDECESSOR.parent/'go2_all_motion_horizon_readout_v1_attempt_001'
ARMS = ('fixed_500ms', 'multi_100_800ms')
STEPS, SEED = 440, 2026092202
digest = prior.previous.original.digest


def save(name, value):
    with (OUTPUT/name).open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def prepare():
    parent = prior.previous.original.parent
    original = json.loads((parent.OUTPUT/'samples.json').read_text())
    rows = {r['sample_id']: r for r in parent.load_training_rows()}
    old = json.loads((PREDECESSOR/'samples.json').read_text())
    old_paths = json.loads((PREDECESSOR/'frame_paths.json').read_text())
    paths, lookup, retention = [], {}, {}

    def add(path):
        path = Path(path).resolve()
        assert not any(p == 'sealed' or p.startswith('sealed_') for p in path.parts)
        assert path.is_file()
        root = path.parent
        if str(root) not in retention:
            marker = root/'depth_retention.json'
            retention[str(root)] = json.loads(marker.read_text()) if marker.exists() else None
        key = str(path)
        if key not in lookup:
            lookup[key] = len(paths)
            paths.append(key)
        return lookup[key]

    original_pairs, original_targets, origins = [], [], []
    for i, sample in enumerate(original):
        row = rows[sample['sample_id']]
        assert row['available'] and row['data_role'] == 'train'
        targets = row['targets']
        if not all(t['motion_valid'] and t['future_image_valid'] and
                   t['contact_valid'] and t['contact'] == 0 for t in targets):
            continue
        assert len(targets) == 8
        root = parent.ROOTS[row['source']]/row['trial']
        frame = sample['frame']
        pairs, motions = [], []
        for h, target in enumerate(targets, 1):
            assert target['offset_ns'] == h*100_000_000
            assert target['future_observation_index'] == frame+h
            pairs.append([add(root/f'rgb_{frame:04d}.png'), add(root/f'rgb_{frame+h:04d}.png')])
            motions.append(target['motion'])
        np.testing.assert_allclose(motions[4], old['old_targets'][i], rtol=0, atol=1e-7)
        original_pairs.append(pairs)
        original_targets.append(motions)
        origins.append(dict(sample_id=sample['sample_id'], source=row['source'],
            trial=row['trial'], frame=frame, data_role='train'))
    assert len(origins) == 3158
    heading_pairs = [[[add(old_paths[a]), add(old_paths[b])] for a, b in row]
                     for row in old['new_pairs']]
    assert len(heading_pairs) == len(old['new_targets']) == 2808
    schedules = {}
    for i, (key, n) in enumerate((('original', 3158), ('heading', 2808))):
        schedules[key] = prior.previous.cycle(n, STEPS*32, SEED+i).reshape(STEPS, 32).tolist()
        horizons = np.tile(np.arange(8), STEPS*32//8)
        np.random.default_rng(SEED+10+i).shuffle(horizons)
        assert np.array_equal(np.bincount(horizons), np.full(8, 1760))
        schedules[key+'_horizons'] = horizons.reshape(STEPS, 32).tolist()
    OUTPUT.mkdir(exist_ok=False)
    save('samples.json', dict(original_pairs=original_pairs, original_targets=original_targets,
        original_origins=origins, heading_pairs=heading_pairs,
        heading_targets=old['new_targets'], heading_origins=old['origins']))
    save('frame_paths.json', paths)
    save('schedule.json', schedules)
    checkpoint = prior.previous.OUTPUT/'mixed_data_final.pt'
    save('plan.json', dict(arms=ARMS, steps=STEPS, batch=64, seed=SEED,
        samples=dict(original=3158, heading=2808), paths=len(paths),
        source_sha256=digest(__file__), initial_checkpoint=str(checkpoint),
        initial_checkpoint_sha256=digest(checkpoint),
        predecessor_samples_sha256=digest(PREDECESSOR/'samples.json'),
        initial_original_samples_sha256=digest(parent.OUTPUT/'samples.json'),
        input_sha256={name:digest(OUTPUT/name) for name in ('samples.json','frame_paths.json','schedule.json')},
        retained_depth_markers=retention, optimizer=dict(name='AdamW', lr=.001, weight_decay=.0001, gradient_clip=1.),
        comparison='same initial mixed head, departure order, 32 original plus 32 heading examples, normalization and 440 updates; fixed500 versus 100--800 ms in both halves',
        checkpoint_selection='fixed final step', encoder_and_predictor_frozen=True,
        no_prospective_maze_training=True, no_navigation=True, automatic_promotion=False,
        feature_storage='pooled FP16 RAM only', device='cpu',
        limitations=['one seed; temporal interval and displacement magnitude change together',
            'common eight-horizon support excludes late recording contexts',
            'readout intervention, not JEPA-objective retraining']))
    print('ALL_MOTION_PREPARED', len(paths), 'paths', flush=True)


def main():
    plan = json.loads((OUTPUT/'plan.json').read_text())
    assert digest(__file__) == plan['source_sha256']
    assert digest(plan['initial_checkpoint']) == plan['initial_checkpoint_sha256']
    for name, sha in plan['input_sha256'].items():
        assert digest(OUTPUT/name) == sha
    assert (OUTPUT/'transfer_plan.json').is_file()
    assert not any((OUTPUT/name).exists() for name in ('process.json','result.json','failure.json'))
    data = json.loads((OUTPUT/'samples.json').read_text())
    paths = json.loads((OUTPUT/'frame_paths.json').read_text())
    schedule = {k:torch.tensor(v, dtype=torch.long) for k,v in json.loads((OUTPUT/'schedule.json').read_text()).items()}
    pairs = {k:torch.tensor(data[k+'_pairs'], dtype=torch.long) for k in ('original','heading')}
    targets = {k:torch.tensor(data[k+'_targets'], dtype=torch.float32) for k in pairs}
    assert psutil.virtual_memory().available > len(paths)*192*1024*2+12*1024**3
    assert shutil.disk_usage(OUTPUT).free > 512*1024**2
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    save('process.json', dict(pid=os.getpid(), create_time=psutil.Process().create_time(), affinity=psutil.Process().cpu_affinity()))
    started = time.monotonic()
    try:
        # Only this process's module globals change; predecessor files stay fixed.
        prior.OUTPUT = OUTPUT
        features = prior.encode(paths)
        models = {a:prior.previous.load('mixed_data').train().requires_grad_(True) for a in ARMS}
        for k,v in models[ARMS[0]].state_dict().items():
            assert torch.equal(v, models[ARMS[1]].state_dict()[k])
        optimizers = {a:torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001) for a,m in models.items()}
        totals = dict.fromkeys(ARMS, 0.)
        history = []
        with (OUTPUT/'progress.jsonl').open('x') as progress:
            for step in range(STEPS):
                for arm, model in models.items():
                    batch_pairs, batch_targets = [], []
                    for group in pairs:
                        idx = schedule[group][step]
                        h = torch.full((32,), 4) if arm == 'fixed_500ms' else schedule[group+'_horizons'][step]
                        batch_pairs.append(pairs[group][idx,h])
                        batch_targets.append(targets[group][idx,h])
                    x = features[torch.cat(batch_pairs)].float()
                    y = torch.cat(batch_targets)
                    opt = optimizers[arm]
                    opt.zero_grad(set_to_none=True)
                    loss = F.mse_loss(model.normalized(x[:,0],x[:,1]),(y-model.target_mean)/model.target_scale)
                    assert torch.isfinite(loss)
                    loss.backward()
                    assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
                    opt.step()
                    totals[arm] += float(loss.detach())
                if (step+1)%20 == 0:
                    row = dict(updates=step+1,train_normalized_mse={a:v/20 for a,v in totals.items()},wall_s=time.monotonic()-started)
                    history.append(row)
                    progress.write(json.dumps(row)+'\n');progress.flush()
                    print('ALL_MOTION_UPDATES',json.dumps(row),flush=True)
                    totals = dict.fromkeys(ARMS,0.)
        hashes = {}
        for arm,model in models.items():
            path = OUTPUT/f'{arm}_final.pt'
            torch.save(dict(model_state_dict=model.state_dict(),updates=STEPS,plan_sha256=digest(OUTPUT/'plan.json')),path)
            hashes[arm] = digest(path)
        save('result.json',dict(status='COMPLETE',steps=STEPS,checkpoint_sha256=hashes,history=history,
            wall_s=time.monotonic()-started,evaluation_pending=True,navigation_tested=False))
        print('ALL_MOTION_COMPLETE',json.dumps(hashes),flush=True)
    except BaseException as error:
        save('failure.json',dict(reason=repr(error),traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else main()
