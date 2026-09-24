"""Repeat the interrupted readout experiment without changing its science."""
import argparse
from types import SimpleNamespace

import torch

from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_maze_view_readout_development as predecessor

original = predecessor.original
evaluation = predecessor.evaluation
OUTPUT = predecessor.OUTPUT.with_name('go2_maze_view_readout_v1_attempt_003')
save = bind(original.save, OUTPUT=OUTPUT)
prepare_fit = bind(original.prepare, OUTPUT=OUTPUT, save=save, __file__=__file__)
fit = bind(original.main, OUTPUT=OUTPUT, save=save,
           bind=predecessor.training_bind, __file__=__file__)
interface = SimpleNamespace(OUTPUT=OUTPUT, ARMS=original.ARMS,
                           STEPS=original.STEPS, prior=original.prior,
                           digest=original.digest)
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
        names = ('samples.json', 'frame_paths.json', 'schedule.json', 'transfer_targets.json')
        identities = {name: original.digest(OUTPUT/name) for name in names}
        assert all(identity == original.digest(predecessor.OUTPUT/name)
                   for name, identity in identities.items())
        save('interruption_recovery.json', dict(predecessor=str(predecessor.OUTPUT),
            interruption_sha256=original.digest(predecessor.OUTPUT/'interruption_record.json'),
            predecessor_source_sha256=original.digest(predecessor.__file__),
            unchanged_input_sha256=identities, scientific_settings_unchanged=True))
    else:
        fit()
        evaluate()
