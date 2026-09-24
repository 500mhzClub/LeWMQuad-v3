"""CPU matched continuation: 500-ms versus 100--800-ms training readout pairs.

Only existing training-role recordings are used. The live cohort continues to
use its unchanged mixed-data checkpoint. Pooled features live in RAM only.
"""
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

from scripts import train_go2_full_heading_readout_development as previous
from scripts.dev_frozen_dense_representation_encoders_v1 import VJepa21Arm
from lewm.dense_visual_motion_readout_development import pool_tokens
from lewm.physical_execution_development import rotation_xyzw

OUTPUT = previous.OUTPUT.parent/'go2_multihorizon_motion_readout_v1_attempt_001'
STEPS, BATCH, SEED = 440, 64, 2026092201
ARMS = ('fixed_500ms', 'multi_100_800ms')


def save(name, value):
    with (OUTPUT/name).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def training_data():
    paths, pairs, targets, old_count, _ = previous.dataset()
    lookup = {str(Path(path).resolve()): i for i, path in enumerate(paths)}

    def add(path):
        resolved = Path(path).resolve()
        assert not any(part == 'sealed' or part.startswith('sealed_')
            or part == 'sealed_test.json' for part in resolved.parts)
        assert resolved.is_file()
        key = str(resolved)
        if key not in lookup:
            lookup[key] = len(paths)
            paths.append(key)
        return lookup[key]

    baseline = {(r['case'], r['frame']): r for r in json.loads(
        (previous.collection.OUTPUT/'samples.json').read_text())}
    new_pairs, new_targets, origins, inputs = [], [], [], []
    for case in range(8):
        root = previous.collection.OUTPUT/f'case_{case:02d}'
        result = json.loads((root/'result.json').read_text())
        spec = json.loads((root/'specification.json').read_text())
        assert result['status'] == 'COMPLETE' and result['data_role'] == spec['data_role'] == 'train'
        assert not result['disallowed_contact']
        # Historical depth is unnecessary; record any intentional retirement.
        retention = root/'depth_retention.json'
        receipt = json.loads(retention.read_text()) if retention.exists() else None
        metadata = root/'in_memory_camera_observations.json'
        frames = json.loads(metadata.read_text())['frames']
        trace = root/'physics_trace.npz'
        with np.load(trace, allow_pickle=False) as data:
            poses = data['base_pose_world'].copy()
            contacts = data['physics_contact'].copy()
        assert not contacts.any()
        inputs.append(dict(case=case, metadata_sha256=previous.original.digest(metadata),
            trace_sha256=previous.original.digest(trace), depth_retention=receipt))
        for frame in range(10, len(frames)-8):
            start = frames[frame]['physical_sample_index']
            origin = poses[start]
            rotation = rotation_xyzw(origin[3:])
            indices, motions = [], []
            for h in range(1, 9):
                assert frames[frame+h]['measured_ns']-frames[frame]['measured_ns'] == h*100_000_000
                end = frames[frame+h]['physical_sample_index']
                future = poses[end]
                relative = rotation.T@rotation_xyzw(future[3:])
                delta = (future[:3]-origin[:3])@rotation
                motion = [float(delta[0]), float(delta[1]), float(np.arctan2(relative[1, 0], relative[0, 0]))]
                indices.append([add(root/f'rgb_{frame:04d}.png'), add(root/f'rgb_{frame+h:04d}.png')])
                motions.append(motion)
            np.testing.assert_allclose(motions[4], baseline[case, frame]['motion'], rtol=0, atol=1e-7)
            new_pairs.append(indices)
            new_targets.append(motions)
            origins.append(dict(case=case, frame=frame, data_role='train'))
    assert len(origins) == 2808
    return (paths, pairs[:old_count], targets[:old_count],
        torch.tensor(new_pairs, dtype=torch.long), torch.tensor(new_targets, dtype=torch.float32), origins, inputs)


@torch.no_grad()
def encode(paths):
    encoder = VJepa21Arm()
    encoder.build(torch.device('cpu'), torch.float32)
    keys = [previous.original.digest(path) for path in paths]
    first = {}
    for i, key in enumerate(keys):
        first.setdefault(key, i)
    features = torch.empty(len(paths), 192, 1024, dtype=torch.float16)
    started = time.monotonic()
    unique = list(first.values())
    for j, i in enumerate(unique):
        pixels = encoder.preprocess(str(paths[i]))[None]
        tokens = F.layer_norm(encoder.tokens(pixels).float(), (1024,))
        assert torch.isfinite(tokens).all()
        features[i] = pool_tokens(tokens)[0].half()
        if (j+1)%32 == 0 or j+1 == len(unique):
            print('MULTIHORIZON_FEATURES', j+1, len(unique), round(time.monotonic()-started, 1), flush=True)
    for i, key in enumerate(keys):
        if first[key] != i:
            features[i].copy_(features[first[key]])
    save('encoding.json', dict(paths=len(paths), unique_images=len(unique),
        seconds=time.monotonic()-started, image_sha256=keys, device='cpu', tensors_saved=False))
    return features


def main():
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    paths, old_pairs, old_targets, new_pairs, new_targets, origins, inputs = training_data()
    cache_bytes = len(paths)*192*1024*2
    assert psutil.virtual_memory().available > cache_bytes+12*1024**3
    assert shutil.disk_usage(OUTPUT.parent).free > 512*1024**2
    OUTPUT.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        old_schedule = previous.cycle(len(old_pairs), STEPS*32, SEED).reshape(STEPS, 32)
        new_schedule = previous.cycle(len(new_pairs), STEPS*32, SEED+1).reshape(STEPS, 32)
        horizons = np.tile(np.arange(8), STEPS*32//8)
        np.random.default_rng(SEED+2).shuffle(horizons)
        horizons = horizons.reshape(STEPS, 32)
        assert np.array_equal(np.bincount(horizons.ravel()), np.full(8, 1760))
        checkpoint = previous.OUTPUT/'mixed_data_final.pt'
        save('plan.json', dict(arms=ARMS, steps=STEPS, batch=BATCH, seed=SEED,
            old_training_samples=len(old_pairs), new_training_origins=len(new_pairs),
            new_target_horizons_ms=list(range(100, 801, 100)),
            shared_old_500ms_examples_per_batch=32, shared_new_origins_per_batch=32,
            source_sha256=previous.original.digest(__file__),
            initial_checkpoint=str(checkpoint), initial_checkpoint_sha256=previous.original.digest(checkpoint),
            training_inputs=inputs, paths=len(paths), feature_cache_bytes=cache_bytes,
            device='cpu', affinity=psutil.Process().cpu_affinity(),
            encoder_and_predictor_frozen=True, original_target_normalization_retained=True,
            optimizer=dict(name='AdamW', lr=.001, weight_decay=.0001, gradient_clip=1.),
            loss='MSE in original training-standardized body-relative XY/yaw',
            checkpoint_selection='fixed final step; no development-score selection',
            hypothesis='training at multiple horizons improves motion decoding outside 500 ms',
            matched_control='same starting checkpoint, old samples, new departure contexts, optimizer and update budget',
            evaluation='fixed final checkpoints on retained exposed development diagnostics after fitting; no automatic promotion',
            prospective_maze_inputs_used=False, no_navigation=True, tensors_saved=False,
            limitations=['one training seed; existing training geometries only',
                'varies future-image interval and associated motion magnitude together',
                'readout training intervention, not JEPA-objective or predictor retraining']))
        save('process.json', dict(pid=os.getpid(), create_time=psutil.Process().create_time()))
        save('frame_paths.json', paths)
        save('samples.json', dict(origins=origins, old_pairs=old_pairs.tolist(), old_targets=old_targets.tolist(),
            new_pairs=new_pairs.tolist(), new_targets=new_targets.tolist()))
        save('schedule.json', dict(old=old_schedule.tolist(), new=new_schedule.tolist(),
            multi_horizon_indices=horizons.tolist()))
        print('MULTIHORIZON_PREPARED', len(paths), cache_bytes, flush=True)
        features = encode(paths)
        models = {arm: previous.load('mixed_data').train().requires_grad_(True) for arm in ARMS}
        for key, value in models[ARMS[0]].state_dict().items():
            assert torch.equal(value, models[ARMS[1]].state_dict()[key])
        optimizers = {arm: torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            for arm, model in models.items()}
        totals = dict.fromkeys(ARMS, 0.)
        history = []
        with (OUTPUT/'progress.jsonl').open('x') as progress:
            for step in range(STEPS):
                old_idx = torch.from_numpy(old_schedule[step])
                new_idx = torch.from_numpy(new_schedule[step])
                for arm in ARMS:
                    h = torch.full((32,), 4) if arm == 'fixed_500ms' else torch.from_numpy(horizons[step])
                    pairs = torch.cat((old_pairs[old_idx], new_pairs[new_idx, h]))
                    target = torch.cat((old_targets[old_idx], new_targets[new_idx, h]))
                    x = features[pairs].float()
                    model = models[arm]
                    optimizer = optimizers[arm]
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.mse_loss(model.normalized(x[:, 0], x[:, 1]), (target-model.target_mean)/model.target_scale)
                    assert torch.isfinite(loss)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    assert torch.isfinite(norm)
                    optimizer.step()
                    totals[arm] += float(loss.detach())
                if (step+1)%20 == 0:
                    row = dict(updates=step+1, train_normalized_mse={a: v/20 for a, v in totals.items()},
                        wall_s=time.monotonic()-started)
                    history.append(row)
                    progress.write(json.dumps(row)+'\n')
                    progress.flush()
                    print('MULTIHORIZON_UPDATES', json.dumps(row), flush=True)
                    totals = dict.fromkeys(ARMS, 0.)
        hashes = {}
        for arm, model in models.items():
            path = OUTPUT/f'{arm}_final.pt'
            torch.save(dict(model_state_dict=model.state_dict(), updates=STEPS,
                plan_sha256=previous.original.digest(OUTPUT/'plan.json')), path)
            hashes[arm] = previous.original.digest(path)
        save('result.json', dict(status='COMPLETE', steps=STEPS, checkpoint_sha256=hashes,
            history=history, wall_s=time.monotonic()-started, evaluation_pending=True,
            navigation_tested=False, original_cohort_checkpoints_unchanged=True))
        print('MULTIHORIZON_COMPLETE', json.dumps(hashes), flush=True)
    except BaseException as error:
        save('failure.json', dict(reason=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
