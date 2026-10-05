"""Stage-2 C1 fairness refit (development; Andrew, 3 October 2026: "C1 is refit on the patch data as a fairness check,
using the same mix as the decoder and C4. It runs alongside unchanged C1.").

**Model.** Exactly C1's form (lewm/short_pulse_navigation_runtime_development.py `command_predictions`, fitted by
scripts/fit_go2_local_motion_controls_development.py `fit`): for each horizon h = 100-800 ms, a ridge regression (penalty
1, weighted standardisation) of the residual true motion minus the nominal command integration, on 447 features:
- the prospective commands known up to h (8 x 3, later steps zero);
- the nominal integration at h (3);
- the past command history (420): for each of the four frames t-300 ms .. t, the applied-command record's 15 values
  (scaled by 0.3, 1, 0.5), valid flags and ages, as `past_commands` builds it from the public observations.

**Data.** The stage-2 feature cache's contexts (scripts/build_go2_stage2_feature_cache_development.py `collect`, identical
items; checked against the cache's items.json when it exists). The prospective commands are the executed applied tape;
targets are the cache's physics-true motion (x, y, yaw at 100-800 ms). History features come from each context's
Recording (policy_histories.npz), the arrays the runtime's public observations carry.

**Weights.** The decoder and C4 refit's batch mix (scripts/fit_go2_stage2_decoder_development.py MIX): each train context
is weighted by its expected draw count over that fit's 3,520 x 64 draws (share / cell size x 225,280).

**Validation (`--validate`).** On the stage-2 recordings, the deployed C1 applied to features rebuilt this way, with each
decision's logged prefix and candidate commands, must reproduce the logged `command_history_forecast_xy_yaw`.

**Evaluation.** The deployed C1 and the refit, on the cache's evaluation sets with the executed tape as commands, by movement
type (and edge bin for eval_patch), with the decoder fit's metrics.

Outputs: `<capability root>/stage2_c1_refit_v1/` (command_only.npz in C1's format, result.json).

Usage: fit_go2_stage2_c1_refit_development.py [--validate]
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from lewm import decision_headroom_json_v42_development as output
from lewm.short_pulse_navigation_runtime_development import COMMAND_FIT, command_predictions
from lewm.terminal_translation_pulse_development import command_sequences
from scripts import build_go2_stage2_feature_cache_development as cachebuild
from scripts import fit_go2_stage2_decoder_development as decoder
from scripts.fit_go2_local_motion_controls_development import fit as ridge_fit, nominal

BASE = cachebuild.v1.BASE
OUT = BASE/'stage2_c1_refit_v1'
DRAWS = decoder.UPDATES*64
SETS = {'eval_onpolicy': (5, 8), 'eval_fresh_c3': (8,), 'eval_offline': (5, 8), 'eval_transfer': (5, 7), 'eval_patch': (5, 8)}


def history(recording, frame):
    """past_commands(), from the Recording's per-frame applied-command arrays for frames t-3 .. t."""
    rows = []
    for f in range(frame-3, frame+1):
        now = recording.stamps[f]
        if now != recording.stamps[frame]+(f-frame)*100_000_000:
            raise ValueError('same causal camera history required')
        measured, available = recording.measured[f], recording.available[f]
        if np.any(measured > now) or np.any(available > now):
            raise ValueError('future command observations forbidden')
        age = np.where(measured >= 0, (now-measured)/1e9, 1.5)[:, None]
        rows.append(np.concatenate((np.asarray(recording.values[f], np.float64)/[.3, 1., .5],
                                    recording.valid[f].astype(float), age), axis=1).ravel())
    out = np.concatenate(rows)
    if out.shape != (420,) or not np.isfinite(out).all():
        raise ValueError('finite command history required')
    return out


def features(commands, past):
    commands = np.asarray(commands, float)
    base = nominal(commands)
    x = np.zeros((8, 447))
    for h in range(8):
        known = np.zeros((8, 3))
        known[:h+1] = commands[:h+1]
        x[h] = np.concatenate((known.ravel(), base[h], past))
    return x, base


def load_model(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in ('mean', 'scale', 'bias', 'coefficient')}


def predict(model, x, base):
    """x (N, 8, 447), base (N, 8, 3) -> (N, 8, 3)."""
    return base+np.stack([((x[:, h]-model['mean'][h])/model['scale'][h])@model['coefficient'][h]+model['bias'][h]
                          for h in range(8)], axis=1)


def validate():
    model = load_model(COMMAND_FIT/'command_only.npz')
    worst, n = 0., 0
    for _arm, run in cachebuild.cohort_runs(cachebuild.COHORT)[:4]:
        recording = cachebuild.v1.training_source(str(run), str(cachebuild.REPLAYS/run.name)).recording
        for row in json.loads((run/'planning.json').read_text()):
            if 'selection' not in row:
                continue
            frame = (row['measured_ns']-1_500_000_000)//100_000_000
            if frame < 3:
                continue
            mc = row['motion_correction']
            logged = np.asarray(mc['command_history_forecast_xy_yaw'])
            ours = command_predictions(model, history(recording, frame),
                                       command_sequences(row['committed_prefix'], pulse=bool(mc['terminal_translation_pulse'])))
            worst = max(worst, float(np.abs(ours-logged).max()))
            n += 1
    print(json.dumps(dict(decisions_checked=n, max_abs_difference=worst)))
    if worst > 1e-6:
        raise ValueError('rebuilt C1 features do not reproduce the logged forecasts')


def build(items, sources):
    x = np.zeros((len(items), 8, 447), np.float64)
    base = np.zeros((len(items), 8, 3))
    for i, it in enumerate(items):
        x[i], base[i] = features(it['tape'], history(sources[it['source']].recording, it['frame']))
    return x, base


def main():
    output.install(BASE)
    started = time.monotonic()
    items, sources = cachebuild.collect()
    cached = BASE/'stage2_feature_cache_v1/items.json'
    if cached.exists():
        saved = json.loads(cached.read_text())
        if len(saved) != len(items) or any({k: v for k, v in s.items() if k != 'frame_rows'} != json.loads(json.dumps(i))
                                           for s, i in zip(saved, items)):
            raise ValueError('items differ from the stage-2 cache')
    x, base = build(items, sources)
    targets = np.asarray([it['targets'] for it in items], np.float64)
    train = np.asarray([it['set'] == 'train' for it in items])
    # Expected draws under the decoder mix.
    weights = np.zeros(len(items))
    total = sum(decoder.MIX.values())
    cells = Counter()
    for key, share in decoder.MIX.items():
        g, c = key.split('/')
        members = [i for i, it in enumerate(items) if train[i] and g in ('*', it['group']) and c in ('*', it['category'])]
        if members:
            weights[members] += DRAWS*share/total/len(members)
            cells[key] = len(members)
    valid = np.isfinite(targets).all(axis=2)
    data = dict(x=x[train], residual=(targets-base)[train], valid=valid[train])
    model = ridge_fit(data, weights[train], np.arange(447))
    OUT.mkdir(exist_ok=False)
    np.savez_compressed(OUT/'command_only.npz', **model)
    deployed = load_model(COMMAND_FIT/'command_only.npz')
    report = {}
    for name, horizons in SETS.items():
        idx = np.flatnonzero([it['set'] == name for it in items])
        cats = np.asarray([items[i]['category'] for i in idx])
        bins = np.asarray([items[i].get('edge_bin', '') for i in idx])
        for label, m in (('C1', deployed), ('C1_refit', model)):
            p = predict(m, x[idx], base[idx])
            for h in horizons:
                key = f'{name}@{100*h}ms/{label}'
                report[key] = decoder.fit.metrics(p[:, h-1], targets[idx, h-1].astype(np.float32), cats)
                if name == 'eval_patch':
                    for b in decoder.EDGE_BINS:
                        if (bins == b).any():
                            report[key][f'bin:{b}'] = decoder.fit.metrics(p[bins == b, h-1], targets[idx[bins == b], h-1], cats[bins == b])['all']
    result = dict(name='stage2_c1_refit_v1', model_form='C1 command-history ridge (penalty 1), 447 features, 8 horizons',
                  train_contexts=int(train.sum()), expected_draws=DRAWS, mix=decoder.MIX, cells=dict(cells),
                  deployed_c1_sha256=hashlib.sha256((COMMAND_FIT/'command_only.npz').read_bytes()).hexdigest(),
                  refit_sha256=hashlib.sha256((OUT/'command_only.npz').read_bytes()).hexdigest(), eval=report,
                  wall_s=time.monotonic()-started, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'result.json').write_text(output.dumps(result, indent=1))
    for key in ('eval_onpolicy@800ms', 'eval_patch@800ms'):
        for label in ('C1', 'C1_refit'):
            r = report[f'{key}/{label}']
            print(key, label, ' '.join(f"{c}:{v['median_xy_mm']:.0f}" for c, v in r.items()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--validate', action='store_true')
    a = p.parse_args()
    validate() if a.validate else main()
