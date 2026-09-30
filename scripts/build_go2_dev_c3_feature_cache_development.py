"""Development feature cache for fast C3 decoder and C4 iteration (development mode, 30 Sep 2026).

One pass over every context of interest. Each unique frame is encoded once with the frozen
V-JEPA 2.1 encoder, and two things are stored:
- `frames.f16`: per-frame pooled, layer-normed tokens (N, 192, 1024) in fp16. These are C4's
  inputs and C3's current-frame features.
- `pred.f16`: the frozen action-conditioned predictor's pooled prediction (P, 192, 1024) for
  each (context, horizon) pair, conditioned on the executed applied tape. These are C3's
  readout inputs.
`items.json` describes every context: its set, frame rows, tape, command history, true motion
and movement category. `pairs.json` maps predicted rows to (item, horizon).

Sets:
- **train:** C3-v3's matched data. All old and maze-pool contexts, and a stratified on-policy
  subsample. Two random horizons per context.
- **eval_onpolicy:** every decision of the 6 held-out on-policy C1 missions (closed-loop
  states), at 500 and 800 ms.
- **eval_offline:** the 1,384 held-out rest/turn recording windows, at 500 and 800 ms.
- **eval_transfer:** the transfer-population contexts, at 500 and 700 ms.
- **eval_fresh_c3:** every decision of the 10 C3-v2 fresh-check missions (C3's own closed-loop
  states, rich in rests and turns), at 800 ms.
Predictor calls are batched across contexts, which differs from deployment only by
batch-composition numerics. Validation and sealed sets are not used.
"""
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from lewm import decision_headroom_json_v42_development as output
from lewm.c3v2_offline_pipeline_development import Recording
from lewm.c3v3_onpolicy_data_development import run_recording
from lewm.dense_visual_motion_readout_development import pool_tokens
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner

BASE = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
OUT = BASE/'dev_c3_cache_v1'
TRANSFER = BASE.parent/'go2_maze_view_transfer_v1_attempt_001'
GAP_REPLAYS = BASE/'c3v2_gap_diagnosis_v1/replays'
SEED = 2026093010
ONPOLICY_QUOTA = dict(hold=10**9, rest_start=10**9, turn=3000, cruise=10**9, arc_steady=2000, switch=4000)
BATCH_CONTEXTS = 16


def category(tape, past):
    """tape (8,3) applied [fwd, lateral, yaw]; past (15,3) applied history."""
    tape, past = np.asarray(tape), np.asarray(past)
    fwd, yaw = tape[:, 0] > 0, tape[:, 2] != 0
    if not tape.any():
        return 'hold'
    if not past[-10:].any() and fwd.any():
        return 'rest_start'
    if not fwd.any():
        return 'turn'
    if np.all(tape == tape[0]):
        return 'cruise' if not yaw.any() else 'arc_steady'
    return 'switch'


class Source:
    """A recording plus how to name its frames."""
    def __init__(self, key, recording, frame_file, meta=None):
        self.key, self.recording, self.frame_file, self.meta = key, recording, frame_file, meta


def training_source(directory, replay=None):
    if replay:
        return Source(str(directory), run_recording(directory, replay), lambda i, r=replay: str(Path(r)/'ego_frames'/f'{i:04d}.png'))
    d = Path(directory)
    return Source(str(d), Recording.training_case(d), lambda i, d=d: str(d/f'rgb_{i:04d}.png'))


def collect():
    rng = np.random.default_rng(SEED)
    items, sources = [], {}

    def source_for(key, make):
        if key not in sources:
            sources[key] = make()
        return sources[key]

    def add(set_name, source, frame, tape, targets, horizons, extra=None):
        past = source.recording.values[frame]
        items.append(dict(set=set_name, source=source.key, frame=int(frame), tape=np.asarray(tape, np.float32).tolist(),
                          past=np.asarray(past, np.float32).tolist(), targets=np.asarray(targets, np.float32).tolist(),
                          horizons=list(horizons), category=category(tape, past), **(extra or {})))
    # Training pool.
    rows = json.loads((BASE/'c3v3_data_v1/train_samples.json').read_text())
    onpolicy = defaultdict(list)
    for index, r in enumerate(rows):
        if r['group'] == 'onpolicy':
            source = source_for(r['directory'], lambda r=r: training_source(r['directory'], r['replay']))
            onpolicy[category(source.recording.executed_tape(r['frame']), source.recording.values[r['frame']])].append(index)
    keep = set()
    for name, indices in onpolicy.items():
        chosen = rng.permutation(indices)[:ONPOLICY_QUOTA[name]]
        keep.update(int(i) for i in chosen)
    for index, r in enumerate(rows):
        if r['group'] == 'onpolicy' and index not in keep:
            continue
        source = source_for(r['directory'], lambda r=r: training_source(r['directory'], r.get('replay')))
        h = rng.choice(np.arange(1, 9), size=2, replace=False)
        add('train', source, r['frame'], source.recording.executed_tape(r['frame']), r['targets'], sorted(int(x) for x in h),
            dict(group=r['group'], row=index))
    # Closed-loop held-out C1 decisions.
    for d in json.loads((BASE/'c3v3_data_v1/heldout_onpolicy_decisions.json').read_text()):
        source = source_for(d['directory'], lambda d=d: training_source(d['directory'], d['replay']))
        add('eval_onpolicy', source, d['frame'], d['executed_tape'], d['targets'], [5, 8], dict(run=d['run'], from_rest=d['from_rest']))
    # Offline held-out recordings.
    for r in json.loads((BASE/'c3v2_data_v1/heldout_samples.json').read_text()):
        source = source_for(r['directory'], lambda r=r: training_source(r['directory']))
        add('eval_offline', source, r['frame'], r['applied_tape'], r['targets'], [5, 8], dict(directory=r['directory']))
    # Transfer population (unique contexts; targets filled at 500/700 from the windows).
    windows = json.loads((TRANSFER/'transfer_targets.json').read_text())
    by_context = defaultdict(dict)
    for w in windows:
        by_context[(w['case'], w['frame'])][w['horizon_ms']] = (w['actual'], w['group'], w['maze'])
    for (case, frame), hs in sorted(by_context.items()):
        directory = TRANSFER/f'case_{case:02d}'
        source = source_for(str(directory), lambda d=directory: training_source(d))
        targets = np.full((8, 3), np.nan, np.float32)
        for ms, (actual, group, maze) in hs.items():
            targets[ms//100-1] = actual
        add('eval_transfer', source, frame, source.recording.executed_tape(frame), targets, [5, 7],
            dict(transfer_group=next(iter(hs.values()))[1], maze=next(iter(hs.values()))[2]))
    # C3-v2's own closed-loop states on the fresh-check mazes.
    for run in sorted((BASE/'runs').glob('c3v2_check_C3_chk*_ep0_attempt001')):
        replay = GAP_REPLAYS/run.name
        source = source_for(str(run), lambda run=run, replay=replay: training_source(run, replay))
        meta = json.loads((run/'native/in_memory_camera_observations.json').read_text())['frames']
        for call in json.loads((run/'model_calls.json').read_text()):
            frame = (call['observed_ns']-1_500_000_000)//100_000_000
            if frame < 10 or frame+8 >= len(meta) or not source.recording.valid[frame].all():
                continue
            try:
                tape = source.recording.executed_tape(frame)
            except ValueError:
                continue
            add('eval_fresh_c3', source, frame, tape, source.recording.targets(frame, meta), [8], dict(run=run.name))
    return items, sources


@torch.inference_mode()
def main():
    output.install(BASE)
    OUT.mkdir(exist_ok=False)
    started = time.monotonic()
    items, sources = collect()
    # Frame table.
    frame_rows = {}
    for it in items:
        s = sources[it['source']]
        it['frame_rows'] = []
        for f in (it['frame']-10, it['frame']-5, it['frame']):
            path = s.frame_file(f)
            if path not in frame_rows:
                frame_rows[path] = len(frame_rows)
            it['frame_rows'].append(frame_rows[path])
    pairs = [(i, h) for i, it in enumerate(items) for h in it['horizons']]
    frames = np.lib.format.open_memmap(OUT/'frames.f16.npy', mode='w+', dtype=np.float16, shape=(len(frame_rows), 192, 1024))
    pred = np.lib.format.open_memmap(OUT/'pred.f16.npy', mode='w+', dtype=np.float16, shape=(len(pairs), 192, 1024))
    pair_row = {p: k for k, p in enumerate(pairs)}
    done_frames = np.zeros(len(frame_rows), bool)
    model = owner.source.load_dense_navigation_model('action', readout_arm='maze_view_maze_data')
    device = next(model.predictor.parameters()).device
    by_source = defaultdict(list)
    for i, it in enumerate(items):
        by_source[it['source']].append(i)
    progress = (OUT/'progress.jsonl').open('x')
    count = 0
    for key, indices in by_source.items():
        s = sources[key]
        indices.sort(key=lambda i: items[i]['frame'])
        tokens = {}
        for k in range(0, len(indices), BATCH_CONTEXTS):
            batch = indices[k:k+BATCH_CONTEXTS]
            needed = sorted({f for i in batch for f in (items[i]['frame']-10, items[i]['frame']-5, items[i]['frame'])})
            missing = [f for f in needed if f not in tokens]
            for m in range(0, len(missing), 16):
                chunk = missing[m:m+16]
                out = F.layer_norm(model.encoder.tokens(torch.stack([s.recording.pixels(f) for f in chunk]).to(device)).float(), (1024,))
                pooled = pool_tokens(out).half().cpu().numpy()
                for f, t, p in zip(chunk, out, pooled):
                    tokens[f] = t
                    row = frame_rows[s.frame_file(f)]
                    if not done_frames[row]:
                        frames[row] = p
                        done_frames[row] = True
                    s.recording._pixels.pop(f, None)
            low = min(items[i]['frame'] for i in batch)-10
            for f in [f for f in tokens if f < low]:
                del tokens[f]
            rows_, ctx, act, ctl, hor = [], [], [], [], []
            for i in batch:
                it = items[i]
                c = torch.stack([tokens[it['frame']-10], tokens[it['frame']-5], tokens[it['frame']]])
                control = torch.tensor(np.asarray(it['past'], np.float32)[:, [0, 2]].reshape(3, 5, 2), device=device)
                control = (control-model.control_mean)/model.control_std
                a = torch.tensor(np.asarray(it['tape'], np.float32)[:, [0, 2]], device=device)
                for h in it['horizons']:
                    rows_.append(pair_row[(i, h)])
                    ctx.append(c)
                    act.append(a)
                    ctl.append(control)
                    hor.append(h)
            n = len(rows_)
            o = model.predictor(torch.stack(ctx), torch.stack(act), torch.tensor(hor, dtype=torch.long, device=device),
                                torch.ones(n, 768, dtype=torch.bool, device=device), control=torch.stack(ctl))
            pooled = pool_tokens(F.layer_norm(o.float(), (1024,))).half().cpu().numpy()
            for r, p in zip(rows_, pooled):
                pred[r] = p
            count += len(batch)
            if count % 2000 < len(batch):
                progress.write(json.dumps(dict(contexts=count, total=len(items), wall_s=time.monotonic()-started))+'\n')
                progress.flush()
        s.recording._pixels.clear()
    frames.flush()
    pred.flush()
    assert done_frames.all(), 'every frame cached'
    for it in items:
        it.pop('horizons_done', None)
    owner.save(OUT/'items.json', items)
    owner.save(OUT/'pairs.json', pairs)
    owner.save(OUT/'frame_rows.json', sorted(frame_rows, key=frame_rows.get))
    counts = defaultdict(lambda: defaultdict(int))
    for it in items:
        counts[it['set']][it['category']] += 1
    owner.save(OUT/'result.json', dict(schema='dev_c3_feature_cache.v1', items=len(items), pairs=len(pairs), frames=len(frame_rows),
                                       counts={k: dict(v) for k, v in counts.items()}, wall_s=time.monotonic()-started,
                                       onpolicy_quota=ONPOLICY_QUOTA, seed=SEED))
    print(json.dumps(dict(items=len(items), pairs=len(pairs), frames=len(frame_rows), counts={k: dict(v) for k, v in counts.items()})), flush=True)


if __name__ == '__main__':
    main()
