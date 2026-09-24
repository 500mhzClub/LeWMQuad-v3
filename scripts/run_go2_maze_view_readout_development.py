"""Same readout experiment with the no-grad encoder wrapper bound correctly.

Attempt 001 failed before encoding or fitting. Preserve it and its source;
attempt 002 changes only the Python wrapper binding and output directory.
"""
import argparse
from types import SimpleNamespace

import torch

from lewm.eligible_floor_registration_development import bind
from scripts import train_go2_maze_view_readout_development as original
from scripts import evaluate_go2_all_motion_horizon_readout_development as evaluation

OUTPUT = original.OUTPUT.with_name('go2_maze_view_readout_v1_attempt_002')
save = bind(original.save, OUTPUT=OUTPUT)


def training_bind(function, **replacements):
    if function is original.prior.encode:
        return torch.no_grad()(bind(function.__wrapped__, **replacements))
    return bind(function, **replacements)


prepare_fit = bind(original.prepare, OUTPUT=OUTPUT, save=save, __file__=__file__)
fit = bind(original.main, OUTPUT=OUTPUT, save=save, bind=training_bind, __file__=__file__)
interface = SimpleNamespace(OUTPUT=OUTPUT, ARMS=original.ARMS, STEPS=original.STEPS,
                           prior=original.prior, digest=original.digest)
prepare_evaluation = bind(evaluation.prepare, fit=interface,
    OUTPUT=OUTPUT/'maze00_evaluation', PLAN=OUTPUT/'transfer_plan.json', __file__=__file__)
evaluate = torch.inference_mode()(bind(evaluation.main.__wrapped__, fit=interface,
    OUTPUT=OUTPUT/'maze00_evaluation', PLAN=OUTPUT/'transfer_plan.json', __file__=__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare_fit()
        prepare_evaluation()
        save('wrapper_correction.json', dict(
            predecessor=str(original.OUTPUT),
            failure_sha256=original.digest(original.OUTPUT/'failure.json'),
            predecessor_source_sha256=original.digest(original.__file__),
            evaluation_source_sha256=original.digest(evaluation.__file__),
            correction='Bind undecorated encoder, then restore torch.no_grad.',
            scientific_settings_unchanged=True))
    else:
        fit()
        evaluate()
