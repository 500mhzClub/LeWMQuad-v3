"""Split-test sanity on the offline held-out recordings (gap diagnosis, development data only).

This is the split test's computation (C3 readouts on actual future features and on the
predictor's features) run on 120 held-out rest/turn-recording windows with at least four
forward steps. The windows are drawn with seed 0. It checks the actual-future path in
distribution, where the readouts are known to work, so the closed-loop result can be read as
coverage and not as an artefact of the path. Validation and sealed sets are untouched.
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


@torch.inference_mode()
def main():
    output.install(BASE)
    rows = json.loads((BASE/'c3v2_data_v1/heldout_samples.json').read_text())
    selected = [r for r in rows if (np.asarray(r['applied_tape'])[:, 0] > 0).sum() >= 4]
    rng = np.random.default_rng(0)
    selected = [selected[i] for i in rng.permutation(len(selected))[:120]]
    model, readouts, _ = scorer.load_models()
    device = next(model.predictor.parameters()).device
    cache, out = {}, []
    for r in selected:
        cache.setdefault(r['directory'], Recording.training_case(r['directory']))
        recording, frame = cache[r['directory']], r['frame']
        native = recording.native(frame)
        context = F.layer_norm(model.encoder.tokens(native['pixels'].to(device)).float(), (1024,))[None]
        control = native['past_applied_commands'][:, [0, 2]].reshape(3, 5, 2).to(device)
        control = ((control-model.control_mean)/model.control_std)[None]
        current = pool_tokens(context[:, -1])
        actions = torch.from_numpy(np.asarray(r['applied_tape'], np.float32)[None][:, :, [0, 2]]).to(device)
        predicted = pool_tokens(F.layer_norm(model.predictor(context, actions, torch.full((1,), 8, dtype=torch.long, device=device),
                                                             torch.ones(1, 768, dtype=torch.bool, device=device), control=control).float(), (1024,)))
        actual = pool_tokens(F.layer_norm(model.encoder.tokens(recording.pixels(frame+8)[None].to(device)).float(), (1024,)))
        row = dict(directory=r['directory'], frame=frame, from_rest=bool(not np.asarray(r['past_applied'])[-10:].any()),
                   true_m=float(np.linalg.norm(np.asarray(r['targets'][7][:2]))))
        for name, readout in readouts.items():
            row[f'{name}_on_actual_future_m'] = float(np.linalg.norm(readout(current, actual).float().cpu().numpy()[0][:2]))
            row[f'{name}_on_predicted_m'] = float(np.linalg.norm(readout(current, predicted).float().cpu().numpy()[0][:2]))
        out.append(row)
    summary = {}
    for split, rest in (('from_rest', True), ('moving', False)):
        subset = [o for o in out if o['from_rest'] == rest and o['true_m'] > .01]
        summary[split] = dict(windows=len(subset), median_true_mm=float(np.median([o['true_m'] for o in subset]))*1000,
                              median_ratio={k: float(np.median([o[k+'_m']/o['true_m'] for o in subset]))
                                            for k in ('C3_v1_on_actual_future', 'C3_v1_on_predicted', 'C3_v2_on_actual_future', 'C3_v2_on_predicted')})
    result = dict(schema='c3v2_split_heldout_sanity.v1', label='Development held-out recordings; validation and sealed sets untouched',
                  summary=summary, rows=out, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    owner.save(scorer.ROOT/'split_heldout_sanity.json', result)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
