"""One fixed CPU readout experiment; frozen encoder/predictors, train roles only.

Four equally spaced admitted 500-ms windows per training recording, fixed ridge
penalty, no architecture/regularization search. Features stay in RAM. The old
physical readout and current full navigation mission remain untouched.
"""
from collections import defaultdict
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from lewm.correlation_motion_readout_development import CorrelationMotionReadout, correlation_difference
from lewm.dense_visual_motion_readout_development import pool_tokens
from scripts.dev_frozen_dense_representation_encoders_v1 import VJepa21Arm
from scripts import train_go2_dense_visual_motion_readout_development as previous
from scripts import train_go2_horizon_dense_predictor_development as horizon

OUTPUT = horizon.OUTPUT.parent/'go2_correlation_motion_readout_v1_attempt_001'
PLAN = Path('docs/go2_correlation_motion_readout_plan_2026-09-18.json')
RIDGE = .1


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def selected_training():
    samples, targets = previous.training_data()
    groups = defaultdict(list)
    for i, sample in enumerate(samples):
        groups[sample['source'], sample['trial']].append(i)
    indices = []
    for key in sorted(groups):
        values = sorted(groups[key], key=lambda i:samples[i]['frame'])
        indices.extend(values[j] for j in np.unique(np.linspace(0, len(values)-1, min(4, len(values))).astype(int)))
    return [samples[i] for i in indices], targets[indices]


def load():
    record = json.loads((OUTPUT/'result.json').read_text())
    assert record['status']=='COMPLETE'
    path = OUTPUT/'readout.pt'
    assert previous.digest(path)==record['model_sha256']
    state = torch.load(path, map_location='cpu', weights_only=True)
    return CorrelationMotionReadout(state['coefficient']).eval().requires_grad_(False)


@torch.inference_mode()
def main():
    assert not OUTPUT.exists() and not PLAN.exists()
    torch.set_num_threads(4)
    samples, target = selected_training()
    paths, lookup, pairs = [], {}, []
    for sample in samples:
        directory = previous.parent.ROOTS[sample['source']]/sample['trial']
        pair = []
        for frame in (sample['frame'], sample['frame']+5):
            path = str(directory/f'rgb_{frame:04d}.png')
            if path not in lookup:
                lookup[path] = len(paths)
                paths.append(path)
            pair.append(lookup[path])
        pairs.append(pair)
    plan = dict(schema='correlation_motion_readout.v1', training_samples=len(samples),
        recordings=len({(s['source'], s['trial']) for s in samples}), unique_paths=len(paths),
        selection='four equally spaced admitted 500-ms windows per recording; no target or error selection',
        samples=[{k:s[k] for k in ('sample_id','source','trial','frame')} for s in samples],
        architecture='normalized 192-token current/future cosine correlation minus current/self correlation, flattened -> ridge XY/yaw without intercept',
        ridge=RIDGE, feature_scale='training root mean square per feature, lower bound .001; then divide by sqrt(36864)',
        target_scale='selected-training standard deviation, lower bound .001; no target centering',
        repeated_image_motion_exactly_zero=True, parameters=192*192*3,
        device='cpu', cpu_cores=[4,5,6,7], encoder_precision='float32',
        feature_cache='FP32 pooled tokens in RAM only', planned_feature_bytes=len(paths)*192*1024*4,
        no_encoder_or_predictor_training=True, no_navigation_intervention=True,
        planned_evaluation='true-future decoding on the complete exposed pilot; then predicted-feature decoding if useful; no prospective-maze use',
        source_sha256={p:previous.digest(p) for p in (__file__,'lewm/correlation_motion_readout_development.py')},
        predecessor_readout_sha256=previous.digest(previous.OUTPUT/'readout.pt'),
        target_sha256=__import__('hashlib').sha256(target.tobytes()).hexdigest())
    OUTPUT.mkdir()
    save(PLAN, plan)
    save(OUTPUT/'plan.json', plan)
    save(OUTPUT/'frame_paths.json', paths)
    started = time.monotonic()
    try:
        encoder = VJepa21Arm()
        encoder.build(torch.device('cpu'), torch.float32)
        features = torch.empty(len(paths), 192, 1024)
        with (OUTPUT/'progress.jsonl').open('x') as progress:
            for i, path in enumerate(paths):
                pixels = encoder.preprocess(path)[None]
                tokens = F.layer_norm(encoder.tokens(pixels).float(), (1024,))
                features[i] = pool_tokens(tokens)[0]
                if i%25==0 or i+1==len(paths):
                    row = dict(encoded=i+1, total=len(paths), wall_s=time.monotonic()-started)
                    progress.write(json.dumps(row)+'\n')
                    progress.flush()
                    print('CORRELATION_FEATURES', json.dumps(row), flush=True)
        del encoder
        x = np.empty((len(pairs), 192*192), dtype=np.float32)
        for i, (a, b) in enumerate(pairs):
            x[i] = correlation_difference(features[a:a+1], features[b:b+1]).numpy()[0]
        scale = np.maximum(np.sqrt(np.mean(x.astype(np.float64)**2, axis=0)), .001)
        divisor = scale*np.sqrt(x.shape[1])
        z = x/divisor
        target_scale = np.maximum(target.std(0), .001)
        alpha = np.linalg.solve(z@z.T+RIDGE*np.eye(len(z)), target/target_scale)
        coefficient = (z.T@alpha)*target_scale[None]/divisor[:, None]
        head = CorrelationMotionReadout(coefficient)
        a, b = pairs[0]
        assert torch.equal(head(features[a:a+1], features[a:a+1]), torch.zeros(1,3))
        assert torch.isfinite(head(features[a:a+1], features[b:b+1])).all()
        predictions = x@head.coefficient.numpy()
        errors = predictions-target
        model_path = OUTPUT/'readout.pt'
        with model_path.open('xb') as stream:
            torch.save(head.state_dict(), stream)
        result = dict(status='COMPLETE', training_samples=len(samples), feature_paths=len(paths),
            model_sha256=previous.digest(model_path), wall_s=time.monotonic()-started,
            training_xy_rmse_mm=float(1000*np.sqrt(np.mean(np.sum(errors[:,:2]**2, axis=1)))),
            training_yaw_rmse_deg=float(np.degrees(np.sqrt(np.mean(errors[:,2]**2)))),
            repeated_image_motion_exactly_zero=True, evaluation_used_for_fit=False,
            no_gpu=True, encoder_and_predictor_unchanged=True, navigation_tested=False,
            scope='fixed diagnostic readout intervention; fewer training samples and smaller head than predecessor, not an encoder-objective comparison')
        save(OUTPUT/'result.json', result)
        print('CORRELATION_READOUT_COMPLETE', json.dumps(result), flush=True)
    except BaseException as error:
        save(OUTPUT/'failure.json', dict(reason=repr(error), traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
