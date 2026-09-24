"""Offline temporal-swap yaw diagnosis on the already selected 32 windows."""
import json
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from scripts import evaluate_go2_dense_maze_turn_readout_development as prior

OUTPUT = prior.ROOT/'turn_pair_symmetry_v1'


def save(name, value):
    with (OUTPUT/name).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    source = prior.OUTPUT/'result.json'
    previous = json.loads(source.read_text())
    plan = json.loads((prior.OUTPUT/'plan.json').read_text())
    assert previous['status'] == 'COMPLETE'
    assert prior.fit.original.digest(prior.fit.OUTPUT/'mixed_data_final.pt') == plan['head_sha256']
    # Respect historical replay markers; only retained RGB is needed here.
    retention = {}
    for name in ('depth_retention.json', 'native/depth_retention.json'):
        path = prior.ROOT/name
        if path.exists():
            retention[name] = json.loads(path.read_text())
    frames = sorted({r['frame']+offset for r in previous['rows']
        for offset in (0, r['horizon_ms']//100)})
    assert len(frames) == 96
    OUTPUT.mkdir(exist_ok=False)
    save('plan.json', dict(source_result=str(source),
        source_result_sha256=prior.fit.original.digest(source),
        source_sha256=prior.fit.original.digest(__file__),
        head_sha256=plan['head_sha256'], depth_retention_receipts=retention,
        device='cpu', selection='unchanged previous 32-window sample, both horizons',
        diagnostic='wrapped forward-plus-backward yaw and half wrapped forward-minus-backward yaw',
        no_fitting=True, no_navigation=True, future_images_offline_only=True))
    started = time.monotonic()
    try:
        head = prior.fit.load('mixed_data')
        encoder = prior.VJepa21Arm()
        encoder.build(torch.device('cpu'), torch.float32)
        features = {}
        for i, frame in enumerate(frames):
            pixels = encoder.preprocess(str(prior.ROOT/'native'/f'rgb_{frame:04d}.png'))[None]
            tokens = F.layer_norm(encoder.tokens(pixels).float(), (1024,))
            features[frame] = prior.pool_tokens(tokens)
            if (i+1)%16 == 0:
                print('PAIR_SYMMETRY_ENCODED', i+1, len(frames), round(time.monotonic()-started, 1), flush=True)
        del encoder
        rows = []
        for old in previous['rows']:
            current = features[old['frame']]
            future = features[old['frame']+old['horizon_ms']//100]
            forward = float(head(current, future)[0, 2])
            backward = float(head(future, current)[0, 2])
            np.testing.assert_allclose(forward, old['predictions']['observed_future'][2], rtol=0, atol=1e-6)
            actual = old['actual'][2]
            symmetric = float(prior.wrapped(forward-backward)/2)
            rows.append(dict(frame=old['frame'], group=old['group'], horizon_ms=old['horizon_ms'],
                actual_yaw_rad=actual, forward_yaw_rad=forward, backward_yaw_rad=backward,
                antisymmetric_yaw_rad=symmetric,
                reversal_inconsistency_rad=float(prior.wrapped(forward+backward)),
                forward_error_rad=float(prior.wrapped(forward-actual)),
                antisymmetric_error_rad=float(prior.wrapped(symmetric-actual)),
                forward_wrong_sign=bool(actual*forward < 0),
                antisymmetric_wrong_sign=bool(actual*symmetric < 0)))
        summaries = []
        for horizon in (500, 700):
            for group in sorted({r['group'] for r in rows}):
                selected = [r for r in rows if r['horizon_ms'] == horizon and r['group'] == group]
                summary = dict(horizon_ms=horizon, group=group, windows=len(selected))
                for key in ('forward_error_rad', 'antisymmetric_error_rad', 'reversal_inconsistency_rad'):
                    summary[key.replace('_rad', '_rmse_deg')] = float(np.degrees(np.sqrt(np.mean([r[key]**2 for r in selected]))))
                for key in ('forward_wrong_sign', 'antisymmetric_wrong_sign'):
                    summary[key] = sum(r[key] for r in selected)
                summaries.append(summary)
        result = dict(status='COMPLETE', rows=rows, summaries=summaries, wall_s=time.monotonic()-started,
            limitations=['same post hoc overlapping sample; no independent navigation result',
                'swapped pairs differ from forward-time training inputs',
                'antisymmetry is a diagnostic intervention, not a validated controller',
                'world-heading yaw reverses exactly; body-relative training yaw may differ slightly',
                'no encoder-versus-readout causal separation or predictor intervention'])
        save('result.json', result)
        print('PAIR_SYMMETRY_COMPLETE', json.dumps(summaries), flush=True)
    except BaseException as error:
        save('failure.json', dict(reason=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
