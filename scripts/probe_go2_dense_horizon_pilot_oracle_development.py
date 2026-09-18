"""CPU-only observed-future decoding on the completed native maze pilot.

No new fit or control decisions. All matched 500-ms pilot windows are included;
the active full mission continues unchanged on separate CPU cores and the GPU.
"""
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from lewm.dense_visual_motion_readout_development import pool_tokens
from scripts.dev_frozen_dense_representation_encoders_v1 import VJepa21Arm
from scripts import train_go2_dense_visual_motion_readout_development as motion_fit
from scripts.read_go2_dense_horizon_pilot_diagnostic_development import ROOT

OUTPUT = ROOT/'observed_future_motion_probe'
DIAGNOSTIC = Path('docs/go2_dense_horizon_pilot_motion_diagnostic_2026-09-18.json')


def save(name, value):
    with (OUTPUT/name).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


@torch.inference_mode()
def main():
    assert not OUTPUT.exists()
    rows = [r for r in json.loads(DIAGNOSTIC.read_text())['rows'] if r['horizon_ms']==500]
    assert len(rows)==35
    OUTPUT.mkdir()
    save('plan.json', dict(pilot=str(ROOT), windows=[r['frame'] for r in rows], horizon_ms=500,
        inputs='completed pilot current and actual future RGB only',
        fitted_head_sha256=motion_fit.digest(motion_fit.OUTPUT/'readout.pt'),
        diagnostic_sha256=hashlib.sha256(DIAGNOSTIC.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        device='cpu', cpu_cores=[4, 5, 6, 7], precision='float32', no_training=True,
        purpose='separate predicted-feature error from frozen physical-readout transfer error',
        all_matched_pilot_windows_included=True, true_future_unavailable_online=True))
    started = time.monotonic()
    try:
        torch.set_num_threads(4)
        encoder = VJepa21Arm()
        encoder.build(torch.device('cpu'), torch.float32)
        head = motion_fit.load()
        frames = sorted({f for r in rows for f in (r['frame'], r['frame']+5)})
        features = {}
        for n, frame in enumerate(frames):
            path = ROOT/'native'/f'rgb_{frame:04d}.png'
            tokens = F.layer_norm(encoder.tokens(encoder.preprocess(str(path))[None]).float(), (1024,))
            features[frame] = pool_tokens(tokens)
            if n%10==0 or n==len(frames)-1:
                print('PILOT_ORACLE_ENCODE', n+1, len(frames), 'seconds', round(time.monotonic()-started, 1), flush=True)
        values = []
        for row in rows:
            frame = row['frame']
            current, future = features[frame], features[frame+5]
            predictions = dict(dense=np.asarray(row['dense_xy_yaw']),
                observed_future=head(current, future)[0].numpy(),
                persistence=head(current, current)[0].numpy())
            actual = np.asarray(row['actual_xy_yaw'])
            errors = {}
            for name, prediction in predictions.items():
                delta = prediction-actual
                delta[2] = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
                errors[name] = delta.tolist()
            values.append(dict(frame=frame, action=row['action'], actual=actual.tolist(),
                predictions={k:v.tolist() for k,v in predictions.items()}, errors=errors))
        metrics = {}
        for name in ('dense', 'observed_future', 'persistence'):
            delta = np.asarray([r['errors'][name] for r in values])
            metrics[name] = dict(xy_rmse_mm=float(1000*np.sqrt(np.mean(np.sum(delta[:, :2]**2, axis=1)))),
                yaw_rmse_deg=float(np.degrees(np.sqrt(np.mean(delta[:, 2]**2)))),
                mean_xy_error_mm=(1000*delta[:, :2].mean(0)).tolist())
        result = dict(status='COMPLETE', windows=len(rows), frames_encoded=len(frames), metrics=metrics,
            rows=values, wall_s=time.monotonic()-started, no_gpu=True, no_training=True,
            observed_future_is_unavailable_online=True,
            prediction_vs_oracle_execution_precision='saved GPU float32 dense predictions; CPU float32 oracle/persistence decoding',
            interpretation='Within-exposed-pilot diagnostic of a frozen 500-ms motion head; not a navigation intervention or proof of what information other decoders could recover.')
        save('result.json', result)
        print('PILOT_ORACLE_COMPLETE', json.dumps(metrics), flush=True)
    except BaseException as error:
        save('failure.json', dict(reason=repr(error), traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
