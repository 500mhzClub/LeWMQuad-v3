"""Stage-2 matched refit of the C3 decoder and C4 on the stage-2 feature cache (development; Andrew, 3 and 5 October 2026).

Plan (docs/go2_navigation_dynamics_perturbation_plan_2026-10-02.md, stage 2, "Refit: matched pair, decoder-fix
procedure"): C3's large past-frames decoder and C4 are fitted on identical data with the `p3_large_past_frames` recipe
(past_frames variant, proj 64, hidden 896, depth 1; 3,520 updates each; lr 3e-4), three seeds, the median seed chosen by
the existing rule. The fitting code is `scripts/fit_go2_dev_decoder_development.py`, unchanged; this module points it at
`stage2_feature_cache_v1`, adds the stage-2 groups and adds the held-out patch set to its evaluation.

**Batch mix (64 per update).** The plan's "32 old, 16 maze, 8 on-policy, 8 patch":
- old 32 and maze 16, as before;
- on-policy 8, split evenly over the six movement types, as `p3_large_past_frames` split its 16;
- patch 8, split 6 : 2 between `patch` (approach, entry, on-patch and exit contexts) and `patch_off` (off-patch contexts
  from the same missions), the plan's 12,000 : 4,000 ratio; uniform within each.

**Evaluation sets.** The decoder fix's four (eval_onpolicy, eval_fresh_c3, eval_offline, eval_transfer) plus
`eval_patch` (every usable decision of the held-out stage-2 recordings, at 500 and 800 ms), reported by movement type and,
for eval_patch, also by edge bin (approach, entry, on_patch, exit, off_patch).

**Pre-refit baseline.** `--baseline` scores the deployed checkpoint (`dev_decoder_fits/p3_large_past_frames_s2026093011.pt`,
the C3 readout and its matched C4, as the stage-1 controllers ran) on the same sets.

Outputs go to `<capability root>/stage2_decoder_fits/`.

Usage:
  fit_go2_stage2_decoder_development.py --baseline
  fit_go2_stage2_decoder_development.py --seed SEED
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts import fit_go2_dev_decoder_development as fit

fit.CACHE = fit.BASE/'stage2_feature_cache_v1'
fit.OUT = fit.BASE/'stage2_decoder_fits'
fit.GROUPS = fit.GROUPS+('patch', 'patch_off')
DEPLOYED = fit.BASE/'dev_decoder_fits/p3_large_past_frames_s2026093011.pt'
NAME = 'stage2_p3_large_past_frames'
SIZE = dict(proj=64, hidden=896, depth=1)
UPDATES, LR = 3520, 3e-4
ONPOLICY = dict(zip(fit.CATEGORIES, (1.33, 1.33, 1.33, 1.33, 1.33, 1.35)))
MIX = {'old/*': 32, 'maze/*': 16, **{f'onpolicy/{c}': v for c, v in ONPOLICY.items()}, 'patch/*': 6, 'patch_off/*': 2}
SEEDS = (2026092205, 2026093011, 2026093012)
EDGE_BINS = ('approach', 'entry', 'on_patch', 'exit', 'off_patch')
decoder_fix_evaluate = fit.evaluate


def evaluate(cache, readout, c4, device, extra=None):
    """The decoder fix's sets, plus eval_patch by movement type and by edge bin."""
    report = decoder_fix_evaluate(cache, readout, c4, device)
    readout.eval()
    c4.eval()
    idx_all = [i for i, it in enumerate(cache.items) if it['set'] == 'eval_patch']
    with torch.inference_mode():
        for h in (5, 8):
            preds = {'C3': [], 'C4': []}
            for k in range(0, len(idx_all), 256):
                idx = idx_all[k:k+256]
                feats, pred, tape, control, y = cache.tensors(idx, np.full(len(idx), h), device)
                preds['C3'].append(readout(feats[:, 2], pred, feats[:, 1], feats[:, 0], control).float().cpu().numpy())
                preds['C4'].append(c4(feats, control, tape, torch.full((len(idx),), h, device=device)).float().cpu().numpy())
            true = np.stack([np.asarray(cache.items[i]['targets'], np.float32)[h-1] for i in idx_all])
            cats = np.asarray([cache.items[i]['category'] for i in idx_all])
            bins = np.asarray([cache.items[i]['edge_bin'] for i in idx_all])
            for model_name, chunks in preds.items():
                p = np.concatenate(chunks)
                report[f'eval_patch@{100*h}ms/{model_name}'] = fit.metrics(p, true, cats)
                for b in EDGE_BINS:
                    m = bins == b
                    if m.any():
                        report[f'eval_patch@{100*h}ms/{model_name}'][f'bin:{b}'] = fit.metrics(p[m], true[m], cats[m])['all']
    readout.train()
    c4.train()
    return report


fit.evaluate = evaluate


def baseline():
    """Score the deployed C3 readout and C4 (one checkpoint) on the stage-2 cache."""
    fit.output.install(fit.BASE)
    fit.OUT.mkdir(exist_ok=True)
    device = torch.device('cuda:0')
    cache = fit.Cache()
    started = time.monotonic()
    saved = torch.load(DEPLOYED, map_location='cpu', weights_only=False)
    readout, c4 = fit.build_models(device, 0, saved['variant'], dict(saved['readout_config']))
    readout.load_state_dict(saved['readout'])
    c4.load_state_dict(saved['c4'])
    report = evaluate(cache, readout, c4, device)
    result = dict(name='stage2_baseline_p3_large_past_frames_s2026093011', baseline=True, checkpoint=str(DEPLOYED),
                  checkpoint_sha256=hashlib.sha256(DEPLOYED.read_bytes()).hexdigest(), eval=report, wall_s=time.monotonic()-started,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (fit.OUT/f"{result['name']}.json").write_text(fit.output.dumps(result, indent=1))
    print(fit.summary(result, keys=('eval_onpolicy@800ms', 'eval_patch@800ms')))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', action='store_true')
    p.add_argument('--seed', type=int, choices=SEEDS)
    a = p.parse_args()
    if a.baseline:
        baseline()
    else:
        fit.main(f'{NAME}_s{a.seed}', MIX, {}, UPDATES, UPDATES, LR, a.seed, 'past_frames', SIZE)
