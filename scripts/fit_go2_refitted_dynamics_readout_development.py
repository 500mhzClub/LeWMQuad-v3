"""Adapt matched motion readouts after the fixed latent predictor refits."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from types import SimpleNamespace

from lewm.eligible_floor_registration_development import bind
from lewm.frozen_representation_dynamics_development import install_predictor
from scripts import fit_go2_frozen_motion_readout_development as original
from scripts import train_go2_frozen_representation_dynamics_development as dynamics

OUTPUT = original.BASE/'go2_refitted_dynamics_readout_v1_attempt_001'
PLAN = Path('docs/go2_refitted_dynamics_readout_plan_2026-09-17.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_parent(name, plan):
    # Model architecture and frozen sensory representation are the original;
    # replace exactly the three fitted history/transition modules.
    model = original.load_parent(name, json.loads(original.PLAN.read_text()))
    record = json.loads((dynamics.OUTPUT/name/'result.json').read_text())
    checkpoint = dynamics.OUTPUT/name/'predictor.pt'
    assert record['status'] == 'complete' and digest(checkpoint) == record['predictor_sha256']
    return install_predictor(model, torch.load(checkpoint, map_location='cpu', weights_only=True))


def load_readout(name):
    return bind(original.load_readout, OUTPUT=OUTPUT, PLAN=PLAN, load_parent=load_parent)(name)


def prepare():
    assert not OUTPUT.exists()
    plan = json.loads(original.PLAN.read_text())
    records = {arm:json.loads((dynamics.OUTPUT/arm/'result.json').read_text()) for arm in original.ARMS}
    assert all(r['status'] == 'complete' for r in records.values())
    plan.update(schema='refitted_dynamics_motion_readout_plan.v1',
        change='same training-only ridge readout fit on frozen newly refitted latent dynamics',
        predecessor_readout_plan_sha256=digest(original.PLAN),
        dynamics_plan_sha256=digest(dynamics.PLAN), dynamics_fit_records=records,
        evaluate_same_fixed_branch_population=True, stale_readout_is_not_a_comparator_to_promote=True,
        source_sha256={p:digest(p) for p in (__file__,
            'scripts/fit_go2_frozen_motion_readout_development.py',
            'lewm/frozen_motion_readout_development.py',
            'lewm/frozen_representation_dynamics_development.py')})
    original.write(PLAN, plan); OUTPUT.mkdir()
    print('PREPARED three matched adapted motion readouts', flush=True)


def fit():
    with torch.inference_mode():
        bind(original.fit.__wrapped__, OUTPUT=OUTPUT, PLAN=PLAN,
            load_parent=load_parent, load_readout=load_readout)()


def evaluate():
    import numpy as np
    from scripts import probe_go2_jepa_latent_branch_science_development as probe
    destination = OUTPUT/'branch_evaluation'
    plan_path = Path('docs/go2_refitted_dynamics_readout_evaluation_plan_2026-09-17.json')
    result_path = Path('docs/go2_refitted_dynamics_readout_result_2026-09-17.json')
    plan = json.loads(probe.PLAN.read_text())
    plan.update(fit_plan_sha256=digest(PLAN), fit_result_sha256=digest(OUTPUT/'result.json'),
        dynamics_plan_sha256=digest(dynamics.PLAN), adapted_motion_readout=True)
    original.write(plan_path, plan)
    context = SimpleNamespace(**(vars(original) | dict(PLAN=PLAN, OUTPUT=OUTPUT, load_readout=load_readout)))

    def save(path, value):
        if isinstance(value, dict) and value.get('status') == 'complete' and 'scores' in value:
            value = value | dict(neural_weights_changed=True, observation_encoders_unchanged=True,
                predictor_fit_plan_sha256=digest(dynamics.PLAN),
                motion_readout_adapted_on_original_training_data_only=True,
                readout_fit_plan_sha256=digest(PLAN), prospective_navigation_not_yet_executed=True)
        original.write(path, value)

    with torch.inference_mode():
        bind(probe.run.__wrapped__, OUTPUT=destination, PLAN=plan_path, RESULT=result_path,
            fits=context, save=save)()
    preceding = dynamics.OUTPUT/'branch_evaluation'
    for filename in ('causal_predictions.npz', 'encoded_targets.npz'):
        with np.load(preceding/filename, allow_pickle=False) as old, \
                np.load(destination/filename, allow_pickle=False) as new:
            assert set(old.files) == set(new.files)
            for name in old.files:
                if not name.endswith('_motion'):
                    np.testing.assert_array_equal(old[name], new[name])
    original.write(destination/'latent_and_target_equality.json', dict(
        all_non_motion_predictions_and_encoded_targets_exactly_match_latent_refit=True,
        only_motion_readout_outputs_change=True))
    print('ADAPTED_READOUT_EVALUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true'); group.add_argument('--fit', action='store_true')
    group.add_argument('--evaluate', action='store_true'); args = parser.parse_args()
    if args.prepare: prepare()
    elif args.fit: fit()
    else: evaluate()
