"""Three matched training-only latent fits; no sensory encoder or readout updates."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import cv2
import torch

from lewm.frozen_representation_dynamics_development import (
    MODULES, reset_predictor, predict_encoded, per_context_loss, predictor_state, install_predictor)
from lewm.pulse_timed_training_runner_development import state_digest
from scripts import fit_go2_frozen_motion_readout_development as fits
from scripts import prepare_go2_short_pulse_training_development as data
from scripts.pre_switch_training_data_development import batch

OUTPUT = fits.BASE/'go2_frozen_representation_dynamics_v1_attempt_001'
PLAN = Path('docs/go2_frozen_representation_dynamics_plan_2026-09-17.json')
ARMS = ('jepa', 'supervised_rollout', 'untrained')
SCHEDULE = data.OUTPUT/'schedule.json'
RATE = 1e-3
CORES = dict(jepa=8, supervised_rollout=0, untrained=1)


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_state(model):
    return {k:v for k,v in model.state_dict().items() if k.split('.')[0] not in MODULES}


def prepare():
    assert not OUTPUT.exists()
    schedule = json.loads(SCHEDULE.read_text())
    assert schedule['updates'] == 1200 and schedule['batch_size'] == 6
    plan = dict(schema='frozen_representation_dynamics_plan.v1', arms=ARMS,
        contexts=4694, updates=1200, draws=7200, batch_size=6,
        schedule_sha256=digest(SCHEDULE), representation_plan_sha256=digest(fits.PLAN),
        representation_fit_result_sha256=digest(fits.OUTPUT/'result.json'),
        source_sha256={p:digest(p) for p in (__file__,
            'lewm/frozen_representation_dynamics_development.py')},
        trainable_modules=MODULES, fresh_common_predictor_initialization=True,
        frozen_modules='online/EMA observation encoders, motion/contact heads and command reference',
        objective='mean per-context latent squared error over actual available future observations',
        optimizer='AdamW', learning_rate=RATE, weight_decay=0., gradient_clip_norm=1.,
        ema_updates=False, no_native_motion_loss=True,
        hyperparameter_search=False, checkpoint_selection=False,
        evaluation='same fixed 36 action-branch contexts; all horizons, 800ms primary; training means and centered action effects',
        transfer_labels_used_for_fit=False, native_navigation=False,
        cpu_cores=CORES, concurrent_fits=3, threads_per_fit=1,
        limits=['one seed and exposed development transfer',
            'representation-specific target spaces; raw errors across arms not directly comparable',
            'existing motion readout becomes stale after predictor refit and is not promoted'])
    save(PLAN, plan); OUTPUT.mkdir()
    print('PREPARED three fixed 1200-update latent-only fits', flush=True)


@torch.no_grad()
def encoded_cache(model, rows, raw):
    count = len(rows)
    encoded = torch.empty((count, 4, 32))
    targets = torch.zeros((count, 8, 32))
    blocks = torch.empty((count, 8, 1, 3))
    valid = torch.empty((count, 8, 1), dtype=torch.bool)
    available = torch.empty((count, 8), dtype=torch.bool)
    maximum_cached_forward_difference = 0.
    for start in range(0, count, 16):
        selected = rows[start:start+16]; stop = start+len(selected)
        b = batch(raw, [r['sample_id'] for r in selected]); inp = b['inputs']; tar = b['targets']
        past = inp['observation_history']
        encoded[start:stop] = model.encoder({k:v.flatten(0, 1) for k,v in past.items()}).reshape(len(selected), 4, 32)
        mask = tar['future_valid']; available[start:stop] = mask
        if mask.any():
            targets[start:stop][mask] = model.target({k:v[mask] for k,v in tar['future_observations'].items()})
        blocks[start:stop] = inp['known_action_blocks']; valid[start:stop] = inp['known_action_valid']
        if start == 0:
            actual = model(**inp)['future_latents']
            cached = predict_encoded(model, encoded[start:stop], blocks[start:stop], valid[start:stop])
            torch.testing.assert_close(cached, actual, rtol=0, atol=0)
            maximum_cached_forward_difference = float((cached-actual).abs().max())
        if stop % 512 == 0 or stop == count:
            print('ENCODED', stop, count, flush=True)
    return dict(past=encoded, target=targets, blocks=blocks, valid=valid,
        available=available), maximum_cached_forward_difference


@torch.no_grad()
def weighted_error(model, encoded, weights):
    values = []
    for start in range(0, len(weights), 64):
        s = slice(start, start+64)
        prediction = predict_encoded(model, encoded['past'][s], encoded['blocks'][s], encoded['valid'][s])
        errors, available = per_context_loss(prediction, encoded['target'][s], encoded['available'][s])
        values.append((errors, available))
    errors = torch.cat([r[0] for r in values]); available = torch.cat([r[1] for r in values])
    return float((errors[available]*weights[available]).sum()/weights[available].sum())


def fit(arm):
    plan = json.loads(PLAN.read_text())
    for p,h in plan['source_sha256'].items(): assert digest(p) == h
    assert digest(SCHEDULE) == plan['schedule_sha256']
    assert digest(fits.PLAN) == plan['representation_plan_sha256']
    assert digest(fits.OUTPUT/'result.json') == plan['representation_fit_result_sha256']
    assert sorted(os.sched_getaffinity(0)) == [CORES[arm]]
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    destination = OUTPUT/arm; destination.mkdir(exist_ok=False)
    save(destination/'launch.json', dict(arm=arm, pid=os.getpid(), plan_sha256=digest(PLAN),
        cpu_affinity=sorted(os.sched_getaffinity(0))))
    started = time.monotonic()
    representation_plan = json.loads(fits.PLAN.read_text())
    model = fits.load_parent(arm, representation_plan).eval().requires_grad_(False)
    initial = fits.load_parent('untrained', representation_plan)
    frozen_hash = state_digest(fixed_state(model))
    predictor_initial_hash = state_digest({name+'.'+k:v for name,s in predictor_state(initial).items() for k,v in s.items()})
    schedule = json.loads(SCHEDULE.read_text()); rows = data.load_training_rows()
    assert len(rows) == 4694 and all(r['data_role'] == 'train' for r in rows)
    raw, identities = data.prepare(rows)
    save(destination/'consumed_policy_sha256.json', identities)
    encoded, equality = encoded_cache(model, rows, raw)
    del raw; gc.collect()
    lookup = {row['sample_id']:i for i,row in enumerate(rows)}
    weights = torch.tensor([schedule['context_draw_counts'][r['sample_id']] for r in rows], dtype=torch.float64)
    original_error = weighted_error(model, encoded, weights)
    parameters = reset_predictor(model, initial); del initial
    optimizer = torch.optim.AdamW(parameters, lr=RATE, weight_decay=0.)
    with (destination/'updates.jsonl').open('x') as ledger:
        for step, identifiers in enumerate(schedule['batches'], 1):
            idx = [lookup[i] for i in identifiers]
            optimizer.zero_grad(set_to_none=True)
            prediction = predict_encoded(model, encoded['past'][idx], encoded['blocks'][idx], encoded['valid'][idx])
            errors, available = per_context_loss(prediction, encoded['target'][idx], encoded['available'][idx])
            assert available.any()
            loss = errors[available].mean()
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            ledger.write(json.dumps(dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm)))+'\n')
            if step == 1 or step % 300 == 0:
                ledger.flush(); print('LATENT_FIT', arm, step, round(float(loss.detach()), 6),
                    round(time.monotonic()-started, 1), flush=True)
    assert step == 1200 and state_digest(fixed_state(model)) == frozen_hash
    final_error = weighted_error(model, encoded, weights)
    state = predictor_state(model)
    checkpoint = destination/'predictor.pt'
    with checkpoint.open('xb') as stream: torch.save(state, stream)
    clone = fits.load_parent(arm, representation_plan)
    install_predictor(clone, torch.load(checkpoint, map_location='cpu', weights_only=True))
    with torch.no_grad():
        args = (encoded['past'][:6], encoded['blocks'][:6], encoded['valid'][:6])
        torch.testing.assert_close(predict_encoded(clone, *args), predict_encoded(model, *args), rtol=0, atol=0)
    save(destination/'result.json', dict(status='complete', arm=arm, updates=step,
        original_weighted_training_latent_mse=original_error, refit_weighted_training_latent_mse=final_error,
        frozen_state_sha256=frozen_hash, frozen_state_unchanged=True,
        common_initial_predictor_sha256=predictor_initial_hash,
        cached_forward_maximum_difference=equality, reloaded_predictor_exact=True,
        training_contexts=len(rows), scheduled_draws=int(weights.sum()),
        contexts_without_future_targets=int((~encoded['available'].any(-1)).sum()),
        predictor_sha256=digest(checkpoint), wall_s=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        transfer_evaluated=False, motion_head_refitted=False, navigation_promoted=False))
    print('LATENT_FIT_COMPLETE', arm, original_error, final_error, flush=True)


def load(arm):
    result = json.loads((OUTPUT/arm/'result.json').read_text())
    assert result['status'] == 'complete'
    checkpoint = OUTPUT/arm/'predictor.pt'
    assert digest(checkpoint) == result['predictor_sha256']
    model = fits.load_readout(arm)
    return install_predictor(model, torch.load(checkpoint, map_location='cpu', weights_only=True))


def evaluate():
    import numpy as np
    from types import SimpleNamespace
    from lewm.eligible_floor_registration_development import bind
    from scripts import probe_go2_jepa_latent_branch_science_development as probe
    from scripts import read_go2_jepa_branch_decomposition_development as decomposition
    destination = OUTPUT/'branch_evaluation'
    eval_plan = Path('docs/go2_frozen_representation_dynamics_evaluation_plan_2026-09-17.json')
    eval_result = Path('docs/go2_frozen_representation_dynamics_result_2026-09-17.json')
    records = {arm:json.loads((OUTPUT/arm/'result.json').read_text()) for arm in ARMS}
    assert all(r['status'] == 'complete' for r in records.values())
    assert len({r['common_initial_predictor_sha256'] for r in records.values()}) == 1
    original_plan = json.loads(probe.PLAN.read_text())
    save(eval_plan, original_plan | dict(predictor_fit_plan_sha256=digest(PLAN),
        evaluation_only_after_all_fits=True, motion_readout_stale_after_predictor_refit=True))

    def write_evaluation(path, value):
        if isinstance(value, dict) and value.get('status') == 'complete' and 'scores' in value:
            value = value | dict(neural_weights_changed=True, observation_encoders_unchanged=True,
                dynamics_fit_records=records, predictor_fit_plan_sha256=digest(PLAN),
                motion_readout_stale_after_predictor_refit=True,
                motion_scores_are_unadapted_head_diagnostic_only=True,
                limits=value['limits']+['old motion readout has not been fitted to changed latent dynamics'])
        save(path, value)

    with torch.inference_mode():
        bind(probe.run.__wrapped__, OUTPUT=destination, PLAN=eval_plan, RESULT=eval_result,
            fits=SimpleNamespace(**(vars(fits) | dict(load_readout=load))), save=write_evaluation)()
    bind(decomposition.main, OUTPUT=destination)()
    with np.load(probe.OUTPUT/'encoded_targets.npz', allow_pickle=False) as old, \
            np.load(destination/'encoded_targets.npz', allow_pickle=False) as new:
        assert set(old.files) == set(new.files)
        for name in old.files: np.testing.assert_array_equal(old[name], new[name])
    save(destination/'representation_equality.json', dict(
        every_encoded_target_and_counterfactual_scene_target_exactly_matches_original=True,
        original_result=str(probe.RESULT), refit_result=str(eval_result)))
    print('FROZEN_REPRESENTATION_EVALUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true'); group.add_argument('--arm', choices=ARMS)
    group.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.evaluate: evaluate()
    else: fit(args.arm)
