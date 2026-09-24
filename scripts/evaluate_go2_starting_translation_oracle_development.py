"""Starting-head actual-future diagnostic on the fixed translation panel."""
import json
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from lewm.dense_visual_motion_readout_development import pool_tokens
from scripts import evaluate_go2_all_motion_horizon_readout_development as fixed


OUTPUT = fixed.fit.OUTPUT / 'starting_translation_oracle_v1'


@torch.inference_mode()
def main():
    plan = json.loads(fixed.PLAN.read_text())
    targets_path = fixed.fit.OUTPUT / 'transfer_targets.json'
    assert fixed.fit.digest(targets_path) == plan['targets_sha256']
    assert fixed.fit.digest(fixed.__file__) == plan['evaluator_sha256']
    training_plan = json.loads((fixed.fit.OUTPUT / 'plan.json').read_text())
    checkpoint = fixed.fit.prior.previous.OUTPUT / 'mixed_data_final.pt'
    assert fixed.fit.digest(checkpoint) == training_plan['initial_checkpoint_sha256']
    receipts = fixed.retention()
    truth = [r for r in json.loads(targets_path.read_text()) if r['group'] == 'translation']
    assert len(truth) == 64
    assert sorted({r['frame'] for r in truth}) == plan['selected']['translation']
    needed = sorted({f for r in truth for f in (r['frame'], r['frame'] + r['horizon_ms'] // 100)})
    OUTPUT.mkdir(exist_ok=False)
    fixed.save(OUTPUT / 'plan.json', dict(
        transfer_plan_sha256=fixed.fit.digest(fixed.PLAN),
        targets_sha256=plan['targets_sha256'], source_sha256=fixed.fit.digest(__file__),
        checkpoint_sha256=training_plan['initial_checkpoint_sha256'], frames=needed,
        selected_translation_frames=plan['selected']['translation'],
        primary_horizon_ms=700, secondary_horizon_ms=500,
        depth_retention_receipts=receipts, cpu_cores=[0, 1, 2, 3],
        no_training=True, no_navigation=True, no_change_to_pending_comparison=True))
    started = time.monotonic()
    torch.set_num_threads(4)
    try:
        head = fixed.fit.prior.previous.load('mixed_data').eval().requires_grad_(False)
        encoder = fixed.VJepa21Arm()
        encoder.build(torch.device('cpu'), torch.float32)
        pooled = {}
        for j, frame in enumerate(needed):
            pixels = encoder.preprocess(str(fixed.ROOT / 'native' / f'rgb_{frame:04d}.png'))[None]
            pooled[frame] = pool_tokens(F.layer_norm(encoder.tokens(pixels).float(), (1024,)))
            if (j + 1) % 16 == 0 or j + 1 == len(needed):
                print('TRANSLATION_ORACLE_FEATURES', j + 1, len(needed), round(time.monotonic() - started, 1), flush=True)
        del encoder
        rows = []
        for target in truth:
            frame, h = target['frame'], target['horizon_ms'] // 100
            actual = np.asarray(target['actual'])
            predictions = dict(starting_mixed_observed_future=head(pooled[frame], pooled[frame + h])[0].numpy(),
                               saved_starting_action=np.asarray(target['saved_starting_action']),
                               command_history=np.asarray(target['command_history']), zero_motion=np.zeros(3))
            errors = {}
            components = {}
            direction = actual[:2] / np.linalg.norm(actual[:2])
            for name, predicted in predictions.items():
                delta = predicted - actual
                delta[2] = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
                errors[name] = delta.tolist()
                components[name] = dict(parallel_error_m=float(delta[:2] @ direction),
                                        transverse_error_m=float(-delta[0] * direction[1] + delta[1] * direction[0]))
            rows.append(target | dict(predictions={k: v.tolist() for k, v in predictions.items()},
                                      errors=errors, components=components))
        summary = {str(h): fixed.metrics([r for r in rows if r['horizon_ms'] == h]) for h in (500, 700)}
        fixed.save(OUTPUT / 'result.json', dict(status='COMPLETE', rows=rows, by_horizon=summary,
            wall_s=time.monotonic() - started, transfer_plan_sha256=fixed.fit.digest(fixed.PLAN),
            limitations=['Actual future images are offline evaluator inputs, unavailable online.',
                         'Selected overlapping windows on one exposed trajectory, not independent trials.',
                         'Saved online action forecasts are reused, not newly inferred in this diagnostic.',
                         'No fitting, calibration, model promotion, or counterfactual navigation.']))
        print('TRANSLATION_ORACLE_COMPLETE', json.dumps(summary), flush=True)
    except BaseException as error:
        fixed.save(OUTPUT / 'failure.json', dict(reason=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
