"""Development fits of the C3 motion readout and the matched C4 from the feature cache (30 Sep 2026).

Development mode: fast iteration on the training mix and loss weighting, judged by prediction
accuracy per movement type. Both models train on the same sampled (context, horizon) stream,
so every C3 variant has a matched C4.

**Mix:** `--mix` gives the batch share of each (group, category) cell as JSON. The groups are
old, maze and onpolicy; the categories are hold, rest_start, turn, cruise, arc_steady and
switch. A '*' matches any. Within a cell, contexts are drawn uniformly.
**Loss weights:** `--weights` gives a per-category multiplier on the loss.
**Decoder inputs:** `--variant` is base, past_frames (a) or history (b); see
`lewm/dev_readout_variants_development.py`.

**Evaluation**, from the cache (the deployed computation, with batch-composition numerics
only). At each set's horizons, by movement category:
- median predicted/true translation (only where true is at least 10 mm);
- median and RMS XY error;
- RMS yaw error;
- median excess translation (predicted minus true magnitude).
Sets: closed-loop held-out C1 decisions (eval_onpolicy), C3-v2's own fresh-check decisions
(eval_fresh_c3, at 800 ms), the offline rest/turn recordings (eval_offline) and the transfer
population (eval_transfer, at 700 ms).
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from lewm import decision_headroom_json_v42_development as output
from lewm.dev_readout_variants_development import ReadoutVariant
from lewm.navigation_capability_supervised_development import DirectMotionPredictor
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
from scripts import train_go2_all_motion_horizon_readout_development as previous

BASE = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
CACHE = BASE/'dev_c3_cache_v1'
OUT = BASE/'dev_decoder_fits'
PREREG = Path('docs/go2_navigation_capability_preregistration_v1_2026-09-25.json')
CATEGORIES = ('hold', 'rest_start', 'turn', 'cruise', 'arc_steady', 'switch')
GROUPS = ('old', 'maze', 'onpolicy')


class Cache:
    def __init__(self):
        self.items = json.loads((CACHE/'items.json').read_text())
        pairs = json.loads((CACHE/'pairs.json').read_text())
        self.pair_row = {(i, h): k for k, (i, h) in enumerate(pairs)}
        self.frames = np.load(CACHE/'frames.f16.npy', mmap_mode='r')
        self.pred = np.load(CACHE/'pred.f16.npy', mmap_mode='r')
        prereg = json.loads(PREREG.read_text())
        stats = json.loads(Path(prereg['harness_v0']['shared_model_and_sensor_bindings']['normalization']['path']).read_text())
        self.mean = np.asarray(stats['control_mean'], np.float32)
        self.std = np.asarray(stats['control_std'], np.float32)

    def control(self, idx):
        past = np.stack([np.asarray(self.items[i]['past'], np.float32)[:, [0, 2]].reshape(3, 5, 2) for i in idx])
        return (past-self.mean)/self.std

    def tensors(self, idx, horizons, device):
        rows = np.asarray([self.items[i]['frame_rows'] for i in idx])
        feats = torch.from_numpy(np.asarray(self.frames[rows.reshape(-1)]).reshape(len(idx), 3, 192, 1024)).to(device).float()
        pred = torch.from_numpy(np.stack([self.pred[self.pair_row[(i, int(h))]] for i, h in zip(idx, horizons)])).to(device).float()
        tape = torch.from_numpy(np.stack([np.asarray(self.items[i]['tape'], np.float32)[:, [0, 2]] for i in idx])).to(device)
        control = torch.from_numpy(self.control(idx)).to(device)
        y = torch.from_numpy(np.stack([np.asarray(self.items[i]['targets'], np.float32)[int(h)-1] for i, h in zip(idx, horizons)])).to(device)
        return feats, pred, tape, control, y


def sampler(cache, mix, rng):
    cells = defaultdict(list)
    for i, it in enumerate(cache.items):
        if it['set'] == 'train':
            cells[(it['group'], it['category'])].append(i)
    plan = []
    for key, share in mix.items():
        g, c = key.split('/')
        members = [i for (gg, cc), v in cells.items() if g in ('*', gg) and c in ('*', cc) for i in v]
        if members and share > 0:
            plan.append((share, np.asarray(members)))
    total = sum(s for s, _ in plan)

    def draw(n):
        counts = rng.multinomial(n, [s/total for s, _ in plan])
        idx = np.concatenate([rng.choice(m, size=k) for (s, m), k in zip(plan, counts) if k])
        h = np.asarray([rng.choice(cache.items[i]['horizons']) for i in idx])
        return idx, h
    return draw, {k: len(v) for k, v in cells.items()}


def build_models(device, seed, variant='base'):
    torch.manual_seed(seed)
    base = previous.prior.previous.load('mixed_data')
    readout = ReadoutVariant(base, past_frames=variant == 'past_frames', history=variant == 'history')
    readout = readout.to(device).train().requires_grad_(True)
    head = torch.load(json.loads(PREREG.read_text())['controllers']['C3']['head_binding']['path'], map_location='cpu', weights_only=False)['model_state_dict']
    c4 = DirectMotionPredictor(head['target_mean'], head['target_scale']).to(device).train()
    return readout, c4


def evaluate(cache, readout, c4, device, extra=None):
    readout.eval()
    c4.eval()
    sets = {'eval_onpolicy': (5, 8), 'eval_fresh_c3': (8,), 'eval_offline': (5, 8), 'eval_transfer': (5, 7)}
    report = {}
    with torch.inference_mode():
        for name, horizons in sets.items():
            idx_all = [i for i, it in enumerate(cache.items) if it['set'] == name]
            for h in horizons:
                preds = {'C3': [], 'C4': []}
                for k in range(0, len(idx_all), 256):
                    idx = idx_all[k:k+256]
                    hs = np.full(len(idx), h)
                    feats, pred, tape, control, y = cache.tensors(idx, hs, device)
                    preds['C3'].append(readout(feats[:, 2], pred, feats[:, 1], feats[:, 0], control).float().cpu().numpy())
                    preds['C4'].append(c4(feats, control, tape, torch.full((len(idx),), h, device=device)).float().cpu().numpy())
                true = np.stack([np.asarray(cache.items[i]['targets'], np.float32)[h-1] for i in idx_all])
                cats = np.asarray([cache.items[i]['category'] for i in idx_all])
                for model_name, chunks in preds.items():
                    p = np.concatenate(chunks)
                    key = f'{name}@{100*h}ms/{model_name}'
                    report[key] = metrics(p, true, cats)
    readout.train()
    c4.train()
    return report


def metrics(p, t, cats):
    out = {}
    for c in ('all',)+CATEGORIES:
        m = np.ones(len(t), bool) if c == 'all' else cats == c
        m &= np.isfinite(t).all(axis=1)
        if not m.any():
            continue
        pt, tt = np.linalg.norm(p[m, :2], axis=1), np.linalg.norm(t[m, :2], axis=1)
        e = np.linalg.norm(p[m, :2]-t[m, :2], axis=1)
        yaw = np.degrees(np.arctan2(np.sin(p[m, 2]-t[m, 2]), np.cos(p[m, 2]-t[m, 2])))
        moving = tt >= .010
        out[c] = dict(n=int(m.sum()), median_ratio=float(np.median(pt[moving]/tt[moving])) if moving.any() else None,
                      median_xy_mm=float(np.median(e))*1000, rmse_xy_mm=float(np.sqrt(np.mean(e**2)))*1000,
                      rmse_yaw_deg=float(np.sqrt(np.mean(yaw**2))), median_excess_mm=float(np.median(pt-tt))*1000,
                      median_true_mm=float(np.median(tt))*1000)
    return out


def main(name, mix, weights, updates_c3, updates_c4, lr, seed, variant):
    output.install(BASE)
    OUT.mkdir(exist_ok=True)
    device = torch.device('cuda:0')
    cache = Cache()
    rng = np.random.default_rng(seed)
    draw, cells = sampler(cache, mix, rng)
    readout, c4 = build_models(device, seed, variant)
    prereg = json.loads(PREREG.read_text())['controllers']['C4']['optimizer']
    opt3 = torch.optim.AdamW(readout.parameters(), lr=lr, weight_decay=1e-4)
    opt4 = torch.optim.AdamW(c4.parameters(), lr=prereg['learning_rate'], weight_decay=prereg['weight_decay'])
    started, history = time.monotonic(), []
    for step in range(max(updates_c3, updates_c4)):
        idx, h = draw(64)
        feats, pred, tape, control, y = cache.tensors(idx, h, device)
        w = torch.tensor([weights.get(cache.items[i]['category'], 1.) for i in idx], device=device)
        row = dict(step=step+1)
        if step < updates_c3:
            opt3.zero_grad(set_to_none=True)
            err = (readout.normalized(feats[:, 2], pred, feats[:, 1], feats[:, 0], control)-(y-readout.target_mean)/readout.target_scale)**2
            loss3 = (err.mean(dim=1)*w).sum()/w.sum()
            loss3.backward()
            torch.nn.utils.clip_grad_norm_(readout.parameters(), 1.)
            opt3.step()
            row['c3_loss'] = float(loss3)
        if step < updates_c4:
            opt4.zero_grad(set_to_none=True)
            err = (c4.normalized(feats, control, tape, torch.from_numpy(h).to(device))-(y-c4.target_mean)/c4.target_scale)**2
            loss4 = (err.mean(dim=1)*w).sum()/w.sum()
            loss4.backward()
            torch.nn.utils.clip_grad_norm_(c4.parameters(), prereg['gradient_clip'])
            opt4.step()
            row['c4_loss'] = float(loss4)
        if (step+1) % 100 == 0:
            history.append(row)
    report = evaluate(cache, readout, c4, device)
    torch.save(dict(readout=readout.state_dict(), c4=c4.state_dict(), mix=mix, weights=weights, variant=variant), OUT/f'{name}.pt')
    result = dict(name=name, variant=variant, mix=mix, weights=weights, updates_c3=updates_c3, updates_c4=updates_c4, lr=lr, seed=seed, cells=
                  {f'{g}/{c}': n for (g, c), n in cells.items()}, history=history, eval=report, wall_s=time.monotonic()-started,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/f'{name}.json').write_text(output.dumps(result, indent=1))
    print(summary(result))


def summary(result, keys=('eval_onpolicy@800ms', 'eval_fresh_c3@800ms', 'eval_offline@800ms', 'eval_transfer@700ms')):
    lines = [f"== {result['name']}  ({result['wall_s']:.0f} s)"]
    for key in keys:
        for model in ('C3', 'C4'):
            r = result['eval'].get(f'{key}/{model}')
            if not r:
                continue
            cells = []
            for c in ('all',)+CATEGORIES:
                if c in r:
                    x = r[c]
                    ratio = '-' if x['median_ratio'] is None else f"{x['median_ratio']:.2f}"
                    cells.append(f"{c}:{x['n']} r{ratio} e{x['median_xy_mm']:.0f}")
            lines.append(f"  {key:22s} {model}  " + '  '.join(cells))
    return '\n'.join(lines)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--mix', default=json.dumps({'old/*': 32, 'maze/*': 16, 'onpolicy/*': 16}))
    p.add_argument('--weights', default='{}')
    p.add_argument('--updates-c3', type=int, default=440)
    p.add_argument('--updates-c4', type=int, default=1760)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=2026092205)
    p.add_argument('--variant', choices=('base', 'past_frames', 'history'), default='base')
    a = p.parse_args()
    main(a.name, json.loads(a.mix), json.loads(a.weights), a.updates_c3, a.updates_c4, a.lr, a.seed, a.variant)
