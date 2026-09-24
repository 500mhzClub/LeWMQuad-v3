"""Fixed exposed-pilot comparison of readouts on true and predicted features.

CPU only. The common 35 executed windows are assessed at 300/500/700 ms,
including the planning interval. Future images are offline oracle inputs only.
"""
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from lewm.dense_visual_motion_readout_development import pool_tokens
from lewm.geometry_progress_pilot_development import ACTIONS
from scripts.dev_frozen_dense_representation_encoders_v1 import VJepa21Arm
from scripts import train_go2_correlation_motion_readout_development as fit
from scripts import train_go2_horizon_dense_predictor_development as dynamics
from scripts import train_go2_dense_visual_motion_readout_development as old_fit
from scripts.read_go2_dense_horizon_pilot_diagnostic_development import ROOT, OUTPUT as DIAGNOSTIC

OUTPUT = fit.OUTPUT/'pilot_evaluation'


def save(name, value):
    fit.save(OUTPUT/name, value)


def metrics(rows):
    result = {}
    for name in rows[0]['errors'] if rows else ():
        values = np.asarray([r['errors'][name] for r in rows])
        result[name] = dict(xy_rmse_mm=float(1000*np.sqrt(np.mean(np.sum(values[:,:2]**2, axis=1)))),
            yaw_rmse_deg=float(np.degrees(np.sqrt(np.mean(values[:,2]**2)))))
    return result


@torch.inference_mode()
def main(*, heads=None, output=None, head_metadata=None):
    global OUTPUT
    if output is not None:
        OUTPUT = Path(output)
    assert not OUTPUT.exists()
    if heads is None:
        heads = dict(correlation=fit.load(), original=old_fit.load())
        head_metadata = dict(correlation=old_fit.digest(fit.OUTPUT/'readout.pt'),
            original=old_fit.digest(old_fit.OUTPUT/'readout.pt'))
    assert 'original' in heads and set(heads)==set(head_metadata)
    torch.set_num_threads(4)
    read = lambda name:json.loads((ROOT/name).read_text())
    diagnostic = json.loads(DIAGNOSTIC.read_text())
    truth = {(r['frame'], r['horizon_ms']):r for r in diagnostic['rows']}
    common = sorted(frame for frame, h in truth if h==700 and (frame,300) in truth and (frame,500) in truth)
    assert len(common)==35
    plans = {r['frame']:r for r in read('planning.json') if 'selection' in r}
    calls = {int((r['observed_ns']-1_500_000_000)//100_000_000):r for r in read('dense_model_calls.json')}
    with np.load(ROOT/'native/policy_histories.npz', allow_pickle=False) as archive:
        past = archive['applied_command_values'].copy()
    stats = json.loads((dynamics.parent.reference.CACHE/'proprio_v1/proprio_norm_stats.json').read_text())
    mean, std = [torch.tensor(stats[k], dtype=torch.float32) for k in ('control_mean','control_std')]
    OUTPUT.mkdir()
    save('plan.json', dict(windows=common, horizons_ms=[300,500,700], pilot=str(ROOT), device='cpu',
        fixed_head_sha256=head_metadata,
        source_sha256=old_fit.digest(__file__), no_training=True, no_navigation=True,
        true_future_images_oracle_only=True, model_inputs='three past native RGB frames and causal applied controls; prospective selected applied action tape',
        comparisons='fixed named heads on action/blind forecasts and observed future; zero-motion control; original-head identity-anchor subtraction',
        scope='same selected executed exposed-pilot windows; no counterfactual or independent navigation outcome'))
    started = time.monotonic()
    try:
        encoder = VJepa21Arm()
        encoder.build(torch.device('cpu'), torch.float32)
        frames = sorted({frame+offset for frame in common for offset in (-10,-5,0,3,5,7)})
        features = {}
        for j, frame in enumerate(frames):
            pixels = encoder.preprocess(str(ROOT/'native'/f'rgb_{frame:04d}.png'))[None]
            features[frame] = F.layer_norm(encoder.tokens(pixels).float(), (1024,))[0]
            if j%25==0 or j+1==len(frames):
                print('READOUT_PILOT_FEATURES', j+1, len(frames), round(time.monotonic()-started,1), flush=True)
        del encoder
        models = {arm:dynamics.load(arm) for arm in dynamics.ARMS}
        rows = []
        for frame in common:
            call = calls[frame]
            selected = ACTIONS.index(plans[frame]['action'])
            actions = torch.tensor(np.asarray(call['applied_commands'])[selected][:,[0,2]], dtype=torch.float32)[None]
            context = torch.stack([features[frame+d] for d in (-10,-5,0)])[None]
            control = (torch.tensor(past[frame][:,[0,2]].reshape(3,5,2), dtype=torch.float32)-mean)/std
            current = pool_tokens(features[frame][None])
            identity = heads['original'](current, current)[0].numpy()
            for horizon in (3,5,7):
                targets = dict(observed_future=features[frame+horizon][None])
                for arm, model in models.items():
                    prediction = model(context, actions, torch.tensor([horizon]),
                        torch.ones(1,768,dtype=torch.bool), control=control[None])
                    targets[arm] = F.layer_norm(prediction.float(), (1024,))
                predictions = {'zero_motion':np.zeros(3)}
                for name, head in heads.items():
                    for arm, tokens in targets.items():
                        value = head(current, pool_tokens(tokens))[0].numpy()
                        predictions[name+'_'+arm] = value
                        if name=='original':
                            predictions['original_anchored_'+arm] = value-identity
                if horizon==5:
                    np.testing.assert_allclose(predictions['original_action'],
                        np.asarray(call['motion_xy_yaw'])[selected,4], rtol=0, atol=2e-5)
                actual = np.asarray(truth[frame,horizon*100]['actual_xy_yaw'])
                errors = {}
                for name, value in predictions.items():
                    delta = value-actual
                    delta[2] = np.arctan2(np.sin(delta[2]),np.cos(delta[2]))
                    errors[name] = delta.tolist()
                rows.append(dict(frame=frame, horizon_ms=horizon*100, actual=actual.tolist(),
                    predictions={k:v.tolist() for k,v in predictions.items()}, errors=errors))
            print('READOUT_PILOT_FORECAST', frame, round(time.monotonic()-started,1), flush=True)
        by = {(r['frame'],r['horizon_ms']):r for r in rows}
        increments = []
        for frame in common:
            a, b = by[frame,300], by[frame,700]
            errors = {}
            for name in a['errors']:
                delta = np.asarray(b['errors'][name])-a['errors'][name]
                delta[2] = np.arctan2(np.sin(delta[2]),np.cos(delta[2]))
                errors[name] = delta.tolist()
            increments.append(dict(frame=frame, errors=errors))
        result = dict(status='COMPLETE', windows=len(common), rows=rows,
            by_horizon={str(h):metrics([r for r in rows if r['horizon_ms']==h]) for h in (300,500,700)},
            commit_interval_300_to_700ms=metrics(increments), increments=increments,
            wall_s=time.monotonic()-started, no_training=True, no_navigation=True,
            original_500ms_predictions_reproduced_within_2e_5=True,
            limitation='exposed selected turn/hold windows; independent navigation and translation remain untested')
        save('result.json', result)
        print('READOUT_PILOT_COMPLETE', json.dumps(dict(by_horizon=result['by_horizon'],
            commit_interval=result['commit_interval_300_to_700ms'])), flush=True)
    except BaseException as error:
        save('failure.json', dict(reason=repr(error), traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
