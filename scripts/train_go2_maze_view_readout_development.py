"""Matched old-data and maze-view continuations of the frozen-feature readout."""
import argparse
from collections import defaultdict
import json
import os
import shutil
import time
import traceback

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from lewm.eligible_floor_registration_development import bind
from scripts import collect_go2_maze_view_training_development as collection
from scripts import train_go2_all_motion_horizon_readout_development as previous

prior = previous.prior
digest = previous.digest
OUTPUT = collection.OUTPUT.parent/'go2_maze_view_readout_v1_attempt_001'
ARMS = ('old_data', 'maze_data')
STEPS, SEED = 440, 2026092205


def save(name, value):
    with (OUTPUT/name).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def prepare():
    completed = json.loads((collection.OUTPUT/'result.json').read_text())
    assert completed['status'] == 'COMPLETE' and completed['training_only']
    assert digest(collection.OUTPUT/'samples.json') == completed['samples_sha256']
    old = json.loads((previous.OUTPUT/'samples.json').read_text())
    paths = json.loads((previous.OUTPUT/'frame_paths.json').read_text())
    lookup = {path: i for i, path in enumerate(paths)}
    groups = defaultdict(list)
    for row in json.loads((collection.OUTPUT/'samples.json').read_text()):
        assert row['data_role'] == 'train'
        groups[row['case'], row['frame']].append(row)
    maze_pairs, maze_targets = [], []
    for rows in groups.values():
        rows.sort(key=lambda r: r['horizon_ms'])
        assert [r['horizon_ms'] for r in rows] == list(range(100, 801, 100))
        pairs = []
        for row in rows:
            pair = []
            for key in ('current_rgb', 'future_rgb'):
                path = row[key]
                if path not in lookup:
                    lookup[path] = len(paths)
                    paths.append(path)
                pair.append(lookup[path])
            pairs.append(pair)
        maze_pairs.append(pairs)
        maze_targets.append([r['motion'] for r in rows])
    assert len(maze_pairs) > 0
    data = dict(old_pairs=old['original_pairs']+old['heading_pairs'],
        old_targets=old['original_targets']+old['heading_targets'],
        maze_pairs=maze_pairs, maze_targets=maze_targets,
        maze_origins=[dict(case=c, frame=f) for c, f in groups])
    schedules = {}
    for i, (key, count) in enumerate((('shared',len(data['old_pairs'])),
            ('other_old',len(data['old_pairs'])), ('maze',len(maze_pairs)))):
        schedules[key] = prior.previous.cycle(count, STEPS*32, SEED+i).reshape(STEPS,32).tolist()
    for i, key in enumerate(('shared_horizons', 'other_horizons')):
        horizons = np.tile(np.arange(8), STEPS*32//8)
        np.random.default_rng(SEED+10+i).shuffle(horizons)
        schedules[key] = horizons.reshape(STEPS,32).tolist()
    OUTPUT.mkdir(exist_ok=False)
    save('samples.json', data)
    save('frame_paths.json', paths)
    save('schedule.json', schedules)
    initial = prior.previous.OUTPUT/'mixed_data_final.pt'
    save('plan.json', dict(arms=ARMS, steps=STEPS, batch=64, seed=SEED,
        source_sha256=digest(__file__), initial_checkpoint=str(initial),
        initial_checkpoint_sha256=digest(initial),
        input_sha256={name:digest(OUTPUT/name) for name in ('samples.json','frame_paths.json','schedule.json')},
        original_samples_sha256=digest(previous.OUTPUT/'samples.json'),
        collection_samples_sha256=completed['samples_sha256'],
        old_contexts=len(data['old_pairs']), maze_contexts=len(maze_pairs), paths=len(paths),
        horizon_ms=list(range(100,801,100)), encoder_and_predictor_frozen=True,
        comparison='32 identical old examples per batch; other 32 old versus new maze examples, with identical horizons; same deployed mixed initialization, architecture, normalization and optimization.',
        optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001,gradient_clip=1.),
        checkpoint_selection='fixed final update', prospective_cohort_excluded=True,
        feature_storage='pooled FP16 RAM only', no_navigation=True,
        automatic_promotion=False, device='cpu',
        limitations=['one training seed', 'geometry, appearance and motion/view coverage change together',
            'exposed development diagnostic evaluation; new prospective navigation required after any promotion']))
    print('MAZE_VIEW_READOUT_PREPARED',len(paths),'images',len(maze_pairs),'new contexts',flush=True)


def main():
    plan = json.loads((OUTPUT/'plan.json').read_text())
    assert digest(__file__) == plan['source_sha256']
    assert digest(plan['initial_checkpoint']) == plan['initial_checkpoint_sha256']
    for name, identity in plan['input_sha256'].items():
        assert digest(OUTPUT/name) == identity
    assert (OUTPUT/'transfer_plan.json').is_file()
    assert not any((OUTPUT/name).exists() for name in ('process.json','result.json','failure.json'))
    data = json.loads((OUTPUT/'samples.json').read_text())
    paths = json.loads((OUTPUT/'frame_paths.json').read_text())
    schedules = {k:torch.tensor(v,dtype=torch.long) for k,v in
                 json.loads((OUTPUT/'schedule.json').read_text()).items()}
    pairs = {k:torch.tensor(data[k+'_pairs'],dtype=torch.long) for k in ('old','maze')}
    targets = {k:torch.tensor(data[k+'_targets'],dtype=torch.float32) for k in pairs}
    assert psutil.virtual_memory().available > len(paths)*192*1024*2+12*1024**3
    assert shutil.disk_usage(OUTPUT).free > 512*1024**2
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    save('process.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),
                            affinity=psutil.Process().cpu_affinity()))
    started = time.monotonic()
    try:
        features = bind(prior.encode, save=save)(paths)
        models = {arm:prior.previous.load('mixed_data').train().requires_grad_(True) for arm in ARMS}
        for key,value in models[ARMS[0]].state_dict().items():
            assert torch.equal(value,models[ARMS[1]].state_dict()[key])
        opts = {arm:torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                for arm,model in models.items()}
        history = []
        totals = dict.fromkeys(ARMS,0.)
        with (OUTPUT/'progress.jsonl').open('x') as progress:
            for step in range(STEPS):
                shared, h = schedules['shared'][step], schedules['shared_horizons'][step]
                other_h = schedules['other_horizons'][step]
                for arm,model in models.items():
                    group = 'old' if arm == 'old_data' else 'maze'
                    idx = schedules['other_old' if group == 'old' else 'maze'][step]
                    batch = torch.cat((pairs['old'][shared,h],pairs[group][idx,other_h]))
                    y = torch.cat((targets['old'][shared,h],targets[group][idx,other_h]))
                    x = features[batch].float()
                    opts[arm].zero_grad(set_to_none=True)
                    loss = F.mse_loss(model.normalized(x[:,0],x[:,1]),
                                      (y-model.target_mean)/model.target_scale)
                    assert torch.isfinite(loss)
                    loss.backward()
                    assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
                    opts[arm].step()
                    totals[arm] += float(loss.detach())
                if (step+1)%20 == 0:
                    row = dict(updates=step+1,train_normalized_mse={k:v/20 for k,v in totals.items()},
                               wall_s=time.monotonic()-started)
                    history.append(row)
                    progress.write(json.dumps(row)+'\n');progress.flush()
                    print('MAZE_VIEW_READOUT_UPDATES',json.dumps(row),flush=True)
                    totals = dict.fromkeys(ARMS,0.)
        hashes = {}
        for arm,model in models.items():
            path = OUTPUT/f'{arm}_final.pt'
            torch.save(dict(model_state_dict=model.state_dict(),updates=STEPS,
                            plan_sha256=digest(OUTPUT/'plan.json')),path)
            hashes[arm] = digest(path)
        save('result.json',dict(status='COMPLETE',steps=STEPS,checkpoint_sha256=hashes,
            history=history,wall_s=time.monotonic()-started,evaluation_pending=True,navigation_tested=False))
        print('MAZE_VIEW_READOUT_COMPLETE',json.dumps(hashes),flush=True)
    except BaseException as error:
        save('failure.json',dict(reason=repr(error),traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else main()
