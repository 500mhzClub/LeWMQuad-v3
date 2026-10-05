"""Offline marker control: paired marked versus unmarked forecasts on the held-out stage-2 contexts (development; Andrew,
5 October 2026).

Each context of `eval_patch` exists twice: in the stage-2 cache (marked frames, as recorded) and in its unmarked twin
(scripts/build_go2_stage2_unmarked_eval_cache_development.py: same physics, same commands and targets, untinted floor).
For the deployed C3/C4 (pre-refit baseline) and each refit seed, at 500 and 800 ms, by edge bin (approach, entry,
on_patch, exit, off_patch):
- median XY error marked and unmarked;
- the median paired difference (unmarked minus marked error, per context), with a 95% bootstrap interval over contexts
  (2,000 resamples, seed fixed), and the same per held-out run.
A gain that depends on the marker shows as unmarked error above marked error in the approach and entry bins for the refit
models, and not (or less) for the baseline, which never saw the marker.

Output: `<capability root>/stage2_decoder_fits/marker_control.json`.

Usage: evaluate_go2_stage2_marker_control_development.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts import fit_go2_stage2_decoder_development as decoder

fit = decoder.fit
MARKED = fit.BASE/'stage2_feature_cache_v1'
UNMARKED = fit.BASE/'stage2_unmarked_eval_cache_v1'
BOOTSTRAP, SEED = 2000, 2026100520


def load_cache(path):
    fit.CACHE = path
    return fit.Cache()


def predictions(cache, idx, readout, c4, device, h):
    out = {'C3': [], 'C4': []}
    with torch.inference_mode():
        for k in range(0, len(idx), 256):
            chunk = idx[k:k+256]
            feats, pred, tape, control, y = cache.tensors(chunk, np.full(len(chunk), h), device)
            out['C3'].append(readout(feats[:, 2], pred, feats[:, 1], feats[:, 0], control).float().cpu().numpy())
            out['C4'].append(c4(feats, control, tape, torch.full((len(chunk),), h, device=device)).float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def load_models(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    readout, c4 = fit.build_models(device, 0, saved['variant'], dict(saved['readout_config']))
    readout.load_state_dict(saved['readout'])
    c4.load_state_dict(saved['c4'])
    return readout.eval(), c4.eval()


def interval(values, rng):
    if len(values) == 0:
        return None
    draws = rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))
    medians = np.median(values[draws], axis=1)
    return [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))]


def main():
    fit.output.install(fit.BASE)
    device = torch.device('cuda:0')
    marked, unmarked = load_cache(MARKED), load_cache(UNMARKED)
    m_idx = [i for i, it in enumerate(marked.items) if it['set'] == 'eval_patch']
    u_idx = list(range(len(unmarked.items)))
    if [(marked.items[i]['source'], marked.items[i]['frame']) for i in m_idx] != [(it['source'], it['frame']) for it in unmarked.items]:
        raise ValueError('marked and unmarked contexts are not paired')
    items = [marked.items[i] for i in m_idx]
    bins = np.asarray([it['edge_bin'] for it in items])
    runs = np.asarray([Path(it['source']).name for it in items])
    checkpoints = {'baseline': decoder.DEPLOYED, **{f'refit_s{s}': fit.OUT/f'{decoder.NAME}_s{s}.pt' for s in decoder.SEEDS}}
    report = {}
    for name, path in checkpoints.items():
        if not path.exists():
            continue
        readout, c4 = load_models(path, device)
        for h in (5, 8):
            true = np.stack([np.asarray(it['targets'], np.float32)[h-1] for it in items])
            pm, pu = predictions(marked, m_idx, readout, c4, device, h), predictions(unmarked, u_idx, readout, c4, device, h)
            for model in ('C3', 'C4'):
                em = np.linalg.norm(pm[model][:, :2]-true[:, :2], axis=1)*1000
                eu = np.linalg.norm(pu[model][:, :2]-true[:, :2], axis=1)*1000
                rng = np.random.default_rng(SEED)
                cell = {}
                for b in decoder.EDGE_BINS+('all',):
                    m = np.ones(len(items), bool) if b == 'all' else bins == b
                    d = eu[m]-em[m]
                    cell[b] = dict(n=int(m.sum()), median_marked_mm=float(np.median(em[m])), median_unmarked_mm=float(np.median(eu[m])),
                                   median_paired_difference_mm=float(np.median(d)), bootstrap_95=interval(d, rng),
                                   per_run={r: float(np.median(d[runs[m] == r])) for r in sorted(set(runs[m]))})
                report[f'{name}/{model}@{100*h}ms'] = cell
    result = dict(name='stage2_marker_control', contexts=len(items), checkpoints={k: str(v) for k, v in checkpoints.items() if v.exists()},
                  report=report, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (fit.OUT/'marker_control.json').write_text(fit.output.dumps(result, indent=1))
    for key, cell in report.items():
        if key.endswith('@800ms'):
            print(key, '  '.join(f"{b}: {c['median_marked_mm']:.0f}->{c['median_unmarked_mm']:.0f} (d {c['median_paired_difference_mm']:+.1f})"
                                 for b, c in cell.items()))


if __name__ == '__main__':
    main()
