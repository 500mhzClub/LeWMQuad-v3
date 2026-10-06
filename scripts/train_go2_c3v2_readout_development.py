"""C3-v2 readout fit on predicted-future features (pre-declared 29 Sep 2026, §3).

Same architecture, initialisation (mixed_data_final), optimiser, 440 updates, batch 64
(32 old + 32 maze-pool) and schedule procedure as the v1 maze-data readout. Only the training
data changes: the maze pool adds the new fit-split rest/turn contexts, and the readout's
future input is the frozen predictor's pooled prediction for the executed applied tape
(C3's run-time computation) instead of the actual future frame's features.
Training features encode each unique frame once per recording in small batches; the effect
of batch composition on the encoder output is measured and recorded.
"""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import psutil
import torch
from torch.nn import functional as F

from lewm import decision_headroom_json_v42_development as output
from lewm import navigation_capability_active_wall_development as wall
from lewm.c3v2_offline_pipeline_development import Recording, pooled_features
from lewm.dense_visual_motion_readout_development import pool_tokens
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
from scripts import train_go2_all_motion_horizon_readout_development as previous

prior = previous.prior
BASE = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
DATA = BASE/'c3v2_data_v1'
OUT = BASE/'c3v2_readout_fit_v1'
STEPS, SEED, BATCH_FRAMES = 440, 2026092205, 8


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schedule(n_old, n_pool):
    shared = prior.previous.cycle(n_old, STEPS*32, SEED+0).reshape(STEPS, 32)
    pool = prior.previous.cycle(n_pool, STEPS*32, SEED+2).reshape(STEPS, 32)
    horizons = []
    for i in range(2):
        h = np.tile(np.arange(8), STEPS*32//8)
        np.random.default_rng(SEED+10+i).shuffle(h)
        horizons.append(h.reshape(STEPS, 32))
    return np.asarray(shared), np.asarray(pool), horizons[0], horizons[1]


@torch.inference_mode()
def extract(model, rows, needed, progress):
    """Current and needed predicted pooled features, one encoder pass per unique frame."""
    device = next(model.predictor.parameters()).device
    by_dir = defaultdict(list)
    for index in sorted(needed):
        by_dir[rows[index]['directory']].append(index)
    current, predicted = {}, {}
    done = 0
    started = time.monotonic()
    for directory, indices in by_dir.items():
        recording = Recording.training_case(directory)
        frames = sorted({f for i in indices for f in (rows[i]['frame']-10, rows[i]['frame']-5, rows[i]['frame'])})
        tokens = {}
        for k in range(0, len(frames), BATCH_FRAMES):
            chunk = frames[k:k+BATCH_FRAMES]
            out = F.layer_norm(model.encoder.tokens(torch.stack([recording.pixels(f) for f in chunk]).to(device)).float(), (1024,))
            for f, t in zip(chunk, out):
                tokens[f] = t
        for index in indices:
            frame = rows[index]['frame']
            context = torch.stack([tokens[frame-10], tokens[frame-5], tokens[frame]])[None]
            native = recording.native(frame)
            control = native['past_applied_commands'][:, [0, 2]].reshape(3, 5, 2).to(device)
            control = ((control-model.control_mean)/model.control_std)[None]
            actions = torch.from_numpy(recording.executed_tape(frame)[None][:, :, [0, 2]]).to(device)
            current[index] = pool_tokens(context[:, -1])[0].half().cpu()
            mask = torch.ones(1, 768, dtype=torch.bool, device=device)
            for h in sorted(needed[index]):
                out = model.predictor(context, actions, torch.full((1,), h+1, dtype=torch.long, device=device), mask, control=control)
                predicted[index, h] = pool_tokens(F.layer_norm(out.float(), (1024,)))[0].half().cpu()
            done += 1
            if done % 250 == 0:
                progress.write(json.dumps(dict(stage='features', contexts=done, total=len(needed), wall_s=time.monotonic()-started))+'\n')
                progress.flush()
    return current, predicted


def main():
    output.install(BASE)
    rows = json.loads((DATA/'train_samples.json').read_text())
    old = [i for i, r in enumerate(rows) if r['group'] == 'old']
    pool = [i for i, r in enumerate(rows) if r['group'] == 'maze']
    shared, maze, shared_h, other_h = schedule(len(old), len(pool))
    needed = defaultdict(set)
    for step in range(STEPS):
        for j in range(32):
            needed[old[shared[step, j]]].add(int(shared_h[step, j]))
            needed[pool[maze[step, j]]].add(int(other_h[step, j]))
    initial = prior.previous.OUTPUT/'mixed_data_final.pt'
    OUT.mkdir(exist_ok=False)
    owner.save(OUT/'plan.json', dict(schema='c3v2_readout_fit.v1', predeclaration='docs/go2_navigation_c3v2_readout_fix_predeclaration_2026-09-29.md',
        predeclaration_sha256=digest('docs/go2_navigation_c3v2_readout_fix_predeclaration_2026-09-29.md'),
        source_sha256=digest(__file__), data_sha256={n: digest(DATA/n) for n in ('train_samples.json', 'result.json')},
        initial_checkpoint=str(initial), initial_checkpoint_sha256=digest(initial), steps=STEPS, batch=64, seed=SEED,
        old_contexts=len(old), maze_pool_contexts=len(pool), needed_contexts=len(needed),
        needed_pairs=sum(len(v) for v in needed.values()), future_input='frozen predictor pooled prediction for the executed applied tape',
        optimizer=dict(name='AdamW', lr=.001, weight_decay=.0001, gradient_clip=1.), checkpoint_selection='fixed final update',
        encoder_and_predictor_frozen=True, heldout_used=False))
    owner.save(OUT/'process.json', dict(pid=os.getpid(), created=psutil.Process().create_time()))
    started = time.monotonic()
    try:
        with wall.job(BASE, 'C3-v2 readout fit'), (OUT/'progress.jsonl').open('x') as progress:
            model = owner.source.load_dense_navigation_model('action', readout_arm='maze_view_maze_data')
            # Batch-composition effect: one context through the run-time path versus the batched cache.
            probe = rows[old[0]]
            recording = Recording.training_case(probe['directory'])
            deployed_cur, deployed_pred = pooled_features(model, recording.native(probe['frame']), recording.executed_tape(probe['frame']), horizons=[8])
            cur, pred = extract(model, rows, {old[0]: {7}}, progress)
            batch_effect = dict(current_max_abs=float((deployed_cur[0].cpu()-cur[old[0]].float()).abs().max()),
                                predicted_max_abs=float((deployed_pred[8][0].cpu()-pred[old[0], 7].float()).abs().max()))
            current, predicted = extract(model, rows, needed, progress)
            del model
            torch.cuda.empty_cache()
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            readout = prior.previous.load('mixed_data').train().requires_grad_(True)
            optimizer = torch.optim.AdamW(readout.parameters(), lr=.001, weight_decay=.0001)
            targets = torch.tensor([r['targets'] for r in rows], dtype=torch.float32)
            history, total = [], 0.
            for step in range(STEPS):
                pairs = [(old[shared[step, j]], int(shared_h[step, j])) for j in range(32)]
                pairs += [(pool[maze[step, j]], int(other_h[step, j])) for j in range(32)]
                x_cur = torch.stack([current[i] for i, _ in pairs]).float()
                x_fut = torch.stack([predicted[i, h] for i, h in pairs]).float()
                y = torch.stack([targets[i, h] for i, h in pairs])
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(readout.normalized(x_cur, x_fut), (y-readout.target_mean)/readout.target_scale)
                assert torch.isfinite(loss)
                loss.backward()
                assert torch.isfinite(torch.nn.utils.clip_grad_norm_(readout.parameters(), 1.))
                optimizer.step()
                total += float(loss.detach())
                if (step+1) % 20 == 0:
                    row = dict(stage='fitting', updates=step+1, train_normalized_mse=total/20, wall_s=time.monotonic()-started)
                    history.append(row)
                    progress.write(json.dumps(row)+'\n')
                    progress.flush()
                    total = 0.
            path = OUT/'readout_v2_final.pt'
            with path.open('xb') as stream:
                torch.save(dict(model_state_dict={k: v.detach().cpu() for k, v in readout.state_dict().items()}, updates=STEPS,
                                plan_sha256=digest(OUT/'plan.json'), model_version='C3-v2 readout'), stream)
            owner.save(OUT/'result.json', dict(status='COMPLETE', checkpoint_sha256=digest(path), updates=STEPS, history=history,
                batch_composition_effect=batch_effect, wall_s=time.monotonic()-started, heldout_used=False, navigation_tested=False))
            print(json.dumps(dict(status='COMPLETE', checkpoint_sha256=digest(path), batch_effect=batch_effect, final=history[-1])), flush=True)
    except BaseException as error:
        owner.save(OUT/'failure.json', dict(reason=repr(error), traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
