"""Gap diagnosis, localisation: C3's frozen predictor or its readout? (check mazes only)

This covers the executed full-forward decisions from the like-for-like set (fresh-check
missions, frames from verified replay). Three things are computed:
1. Readout on actual futures. Both C3 readouts decode the encoder's features of the *actual*
   future frame (frame + h, the 100-ms camera frames) in place of the predictor's output.
   If this recovers the true travel, the loss is in the predictor; if not, in the readout.
2. Feature-space check. At 800 ms, the change the predictor predicts, compared with the
   actual change: ||predicted - current|| against ||actual - current||, and the cosine
   between the two change vectors, on pooled layer-normed tokens.
3. The same current-feature path as the pipeline test, so every number is on the deployed
   computation.
Validation and sealed sets are untouched.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from lewm import decision_headroom_json_v42_development as output
from lewm.c3v2_offline_pipeline_development import Recording
from lewm.dense_visual_motion_readout_development import pool_tokens
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
from scripts import score_go2_c3v2_gap_pipeline_development as scorer

BASE = scorer.BASE
ROOT = scorer.ROOT
FORWARD = 1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def main():
    output.install(BASE)
    rows = [r for r in json.loads((ROOT/'pipeline_and_like_for_like.json').read_text())['forward_executed_rows'] if r['forward_steps'] >= 4]
    model, readouts, _ = scorer.load_models()
    device = next(model.predictor.parameters()).device
    cache, out = {}, []
    for r in rows:
        run = BASE/'runs'/r['run']
        if r['run'] not in cache:
            n = run/'native'
            cache[r['run']] = (Recording(n/'policy_histories.npz', n/'policy_observations.json',
                                         lambda i, d=scorer.REPLAYS/r['run']: d/'ego_frames'/f'{i:04d}.png', n/'physics_trace.npz'),
                               {c['observed_ns']: c for c in json.loads((run/'model_calls.json').read_text())})
        recording, calls = cache[r['run']]
        frame = r['frame']
        call = calls[1_500_000_000+100_000_000*frame]
        native = recording.native(frame)
        context = F.layer_norm(model.encoder.tokens(native['pixels'].to(device)).float(), (1024,))[None]
        control = native['past_applied_commands'][:, [0, 2]].reshape(3, 5, 2).to(device)
        control = ((control-model.control_mean)/model.control_std)[None]
        current = pool_tokens(context[:, -1])
        actions = torch.from_numpy(np.asarray(call['applied_commands'][FORWARD], np.float32)[None][:, :, [0, 2]]).to(device)
        mask = torch.ones(1, 768, dtype=torch.bool, device=device)
        predicted = pool_tokens(F.layer_norm(model.predictor(context, actions, torch.full((1,), 8, dtype=torch.long, device=device), mask,
                                                             control=control).float(), (1024,)))
        future = recording.pixels(frame+8)[None].to(device)
        actual = pool_tokens(F.layer_norm(model.encoder.tokens(future).float(), (1024,)))
        row = dict(run=r['run'], frame=frame, from_rest=r['from_rest'], true_800_xy=r['true_800_xy'])
        for name, readout in readouts.items():
            row[f'{name}_on_actual_future'] = readout(current, actual).float().cpu().numpy()[0][:2].tolist()
            row[f'{name}_on_predicted'] = readout(current, predicted).float().cpu().numpy()[0][:2].tolist()
        dp, da = (predicted-current).flatten(), (actual-current).flatten()
        row.update(predicted_change_norm=float(dp.norm()), actual_change_norm=float(da.norm()),
                   change_cosine=float(F.cosine_similarity(dp, da, dim=0)))
        out.append(row)
    summary = {}
    for split, keep in (('from_rest', lambda r: r['from_rest']), ('moving', lambda r: not r['from_rest']), ('all', lambda r: True)):
        subset = [r for r in out if keep(r)]
        t = np.asarray([r['true_800_xy'] for r in subset])
        s = dict(decisions=len(subset))
        for key in ('C3_v1_on_actual_future', 'C3_v1_on_predicted', 'C3_v2_on_actual_future', 'C3_v2_on_predicted'):
            p = np.asarray([r[key] for r in subset])
            s[key] = dict(median_ratio=float(np.median(np.linalg.norm(p, axis=1)/np.linalg.norm(t, axis=1))),
                          median_xy_error_mm=float(np.median(np.linalg.norm(p-t, axis=1)))*1000)
        s['predicted_over_actual_feature_change'] = float(np.median([r['predicted_change_norm']/r['actual_change_norm'] for r in subset]))
        s['change_cosine_median'] = float(np.median([r['change_cosine'] for r in subset]))
        summary[split] = s
    result = dict(schema='c3v2_gap_localisation.v1', label='Diagnosis on check mazes; validation and sealed sets untouched',
                  summary=summary, rows=out, script_sha256=sha(__file__), input_sha256=sha(ROOT/'pipeline_and_like_for_like.json'))
    owner.save(ROOT/'localisation.json', result)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
