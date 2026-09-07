"""Bounded independent-layout development comparison; no navigation or resume.

Execution requires a frozen source-definition digest and the completed original
supervisor result digest. This is not a launcher for partial cohorts or hardware.
"""
import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

from lewm.cumulative_pulse_learning_development import CumulativePulseTrainer
from lewm.independent_pulse_input_ablation_development import VARIANTS
from lewm.independent_pulse_study_runner_development import train_schedule, predict_heads
from lewm.matched_action_hazard_evaluation_development import evaluate_matched_hazards
from lewm.pulse_timed_dataset_development import ROLES
from scripts.cumulative_pulse_snapshot_development import save_snapshot, load_snapshot, config, MAX_BYTES
from scripts.independent_rgb_body_batch_development import BATCHES, output_root
from scripts.independent_rgb_body_study_data_development import load_study, require
from scripts.independent_rgb_body_study_stream_development import AuditedStudyStream
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts, artifact_path
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch
from scripts.run_go2_independent_rgb_body_remaining_stages_v1 import OUTPUT as SEQUENCE, PYTHON
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

SOURCE = 'scripts/run_go2_independent_pulse_matched_study_v1.py'
TEST = 'lewm/tests/test_independent_pulse_matched_study_development.py'
PROTOCOL = 'docs/go2_independent_pulse_matched_study_v1_2026-09-07.md'
OUTPUT = BASE / 'go2_independent_pulse_matched_study_v1_attempt_001'
SEQUENCE_LAUNCH = 'e47240aeacd1b3af711d2c9d8ba1698b62ad1020982bcd099b47f8ef8c873a26'
SEEDS = (2026091101, 2026091102, 2026091103)
CONDITIONS = ('direct', 'supervised_rollout', 'jepa')
UPDATES = 1200
BATCH_SIZE = 6
INFERENCE_BATCH = 6
LATENT_DIM = 32
BUDGET = 8 * 1024**3
RESERVE = 40 * 1024**3
MAX_METADATA = 64 * 1024**2
ENVIRONMENT = dict(PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
    PYTHONPATH='.:lewm_genesis:lewm_worlds', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def identity(value):
    return hashlib.sha256(encode(value)).hexdigest()


def settings():
    return dict(seeds=list(SEEDS), conditions=list(CONDITIONS), input_variants=list(VARIANTS),
        updates_per_fit=UPDATES, batch_size=BATCH_SIZE, inference_batch_size=INFERENCE_BATCH,
        latent_dim=LATENT_DIM, learning_rate=.001, ema_momentum=.99, position_scale_m=.06,
        maximum_artifact_bytes=BUDGET, minimum_free_bytes=RESERVE,
        torch_version=str(torch.__version__), numpy_version=np.__version__,
        environment=ENVIRONMENT, python=str(PYTHON), cpu_threads=1, deterministic_algorithms=True)


def definition():
    """Read-only prospective source/config binding; no cohort or output access."""
    verify_artifacts(SEQUENCE, {'launch.json': SEQUENCE_LAUNCH})
    old = read_json(SEQUENCE, 'launch.json'); verify_ordered_launch(old)
    result = {k: deepcopy(old[k]) for k in ('input_sha256', 'native_sha256', 'native_scene_sha256',
        'native_geometry_sha256', 'opencv_version', 'opencv_binary_sha256', 'rules')}
    result.update(source_sha256=discover_sources((SOURCE, TEST, PROTOCOL), old['source_sha256']),
        schema='independent_pulse_matched_study.v1', output_root=str(OUTPUT),
        sequence_launch_sha256=SEQUENCE_LAUNCH, **settings(),
        snapshot_budget='one final snapshot per fresh fit',
        checkpoint_selection_performed=False, final_evaluation=False,
        navigation_qualified=False, hardware_qualified=False, goal_achieved=False)
    verify_ordered_launch(result)
    return result


def preflight(sequence_result_sha256, definition_sha256):
    require(str(sys.executable) == str(PYTHON) and os.getcwd() == str(ROOT)
        and all(os.environ.get(k) == v for k, v in ENVIRONMENT.items())
        and torch.get_num_threads() == 1 and torch.are_deterministic_algorithms_enabled(),
        'exact CPU interpreter, directory, environment and deterministic mode required')
    validate_root(OUTPUT, must_exist=False)
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), 'exclusive study attempt; no retry/resume')
    d = definition()
    require(identity(d) == definition_sha256, 'reviewed exact source/config definition required')
    require(not (SEQUENCE / 'failure.json').exists(), 'failed supervisor cannot supply the study')
    receipt = {'launch.json': SEQUENCE_LAUNCH, 'result.json': sequence_result_sha256}
    verify_artifacts(SEQUENCE, receipt)
    terminal = read_json(SEQUENCE, 'result.json')
    require(terminal['status'] == 'ALL12_FIXED_RGB_BODY_BATCH_RECEIPTS_VERIFIED'
        and terminal['completed_batches'] == list(BATCHES)
        and set(terminal['receipts']) == set(BATCHES)
        and terminal['planned_layouts'] == 12 and terminal['planned_episodes'] == 1440
        and all(terminal[k] is False for k in ('model_training', 'final_evaluation', 'navigation_qualified', 'goal_achieved')),
        'complete original twelve-layout terminal result required')
    verify_artifacts(SEQUENCE, terminal['output_sha256'])
    study = load_study(terminal['receipts'])
    verify_artifacts(SEQUENCE, receipt)
    require(shutil.disk_usage(BASE).free >= RESERVE + BUDGET, 'full study artifact allowance and reserve required')
    return study, d, receipt


def coverage(study):
    """Predeclared layout/action gates and separate joint-supervision accounting."""
    view = study.evaluation; report = {}; reasons = []
    for role in ROLES:
        population = view.population(role); planned = population['layouts']
        counts = view.dataset.coverage(role)
        for layout in planned:
            for action in range(6):
                if counts.get(layout, {}).get(str(action), 0) == 0:
                    reasons.append(f'{role}:{layout}:missing_action_{action}')
        rows = []
        for i, window in enumerate(view.dataset.windows):
            meta = view.dataset.episode_roles[window['condition']]
            if meta['role'] != role or not window['history_ready']: continue
            target = view.dataset._targets[i]
            fv = np.asarray([r['future_valid'] for r in window['targets']], bool)
            mv = target['motion_valid'].numpy(); cv = target['contact_valid'].numpy()
            positive = cv & (target['contact'].numpy() == 1)
            rows.append(dict(condition=window['condition'], layout_id=meta['layout_id'],
                action_index=window['action_index'], active=int((target['target_offsets_ns'] > 0).sum()),
                motion=int(mv.sum()), contact=int(cv.sum()), contact_positive=int(positive.sum()),
                future_image=int(fv.sum()), positive_with_future_image=int((positive & fv).sum()),
                positive_without_future_image=int((positive & ~fv).sum()),
                motion_with_future_image=int((mv & fv).sum())))
        fields = ('active', 'motion', 'contact', 'contact_positive', 'future_image',
            'positive_with_future_image', 'positive_without_future_image', 'motion_with_future_image')
        totals = {k: sum(r[k] for r in rows) for k in fields}
        hazards = evaluate_matched_hazards(view, study.prefixes, {}, role=role)
        # Missing/censored action groups stay in the coverage report. Unequal or
        # unavailable original prefixes invalidate the matched-design claim.
        if any(r['unavailable_prefix_actions'] or r['unequal_prefix_actions'] for r in hazards['groups']):
            reasons.append(role + ':unmatched_original_histories')
        report[role] = dict(population=population, action_coverage=counts, episodes=rows,
            layouts={layout: {k: sum(r[k] for r in rows if r['layout_id'] == layout) for k in fields}
                for layout in planned}, totals=totals, matched_contact_coverage=hazards,
            observed_contact_contrast_available=hazards['contrastive_groups'] > 0)
    return dict(eligible_to_fit=not reasons, blocking_reasons=reasons, roles=report,
        no_contact_contrast_is_not_positive_hazard_evidence=True, final_evaluation=False,
        navigation_qualified=False, goal_achieved=False)


class Artifacts:
    """Bounded exclusive writes with durable per-update ledgers; no discovery."""
    def __init__(self, output):
        self.output = validate_root(output); self.hashes = {}; self.used = 0

    def capacity(self, amount):
        require(self.used + amount <= BUDGET and shutil.disk_usage(BASE).free >= RESERVE + amount,
            'study artifact allowance or storage reserve exhausted')

    def path(self, name):
        import re
        require(type(name) is str and re.fullmatch('[a-z][a-z0-9_]*[.](json|jsonl|npz|pt)', name)
            and not name.startswith('sealed_'), 'ordinary explicit study artifact filename required')
        p = self.output / name
        require(not p.exists() and not p.is_symlink(), 'exclusive study artifact required')
        return p

    def save_bytes(self, name, raw):
        require(len(raw) <= MAX_METADATA, 'bounded individual study metadata/array file required')
        self.capacity(len(raw)); p = self.path(name)
        with p.open('xb') as target:
            target.write(raw); target.flush(); os.fsync(target.fileno())
        self.used += len(raw); self.hashes[name] = hashlib.sha256(raw).hexdigest()
        verify_artifacts(self.output, {name: self.hashes[name]})

    def save(self, name, value):
        self.save_bytes(name, encode(value))

    def save_predictions(self, name, result):
        arrays = {head: row['prediction'] for head, row in result['heads'].items()}
        arrays['indices'] = next(iter(result['heads'].values()))['indices']
        raw = io.BytesIO(); np.savez_compressed(raw, **arrays)
        self.save_bytes(name, raw.getvalue())
        with np.load(artifact_path(self.output, name), allow_pickle=False) as restored:
            require(set(restored.files) == set(arrays) and all(restored[k].dtype == v.dtype
                and np.array_equal(restored[k], v, equal_nan=True) for k, v in arrays.items()),
                'prediction-array reload mismatch')
        verify_artifacts(self.output, {name: self.hashes[name]})

    def snapshot(self, name, trainer, binding):
        self.capacity(MAX_BYTES); self.path(name)
        row = save_snapshot(self.output, name, trainer, binding)
        self.used += row['bytes']; self.hashes[name] = row['sha256']
        clone = load_snapshot(self.output, name, sha256=row['sha256'],
            expected_binding=binding, expected_config=config(trainer))
        return row, clone


def baseline_heads(view, empirical, role):
    data = view.arrays(role)
    prediction, missing = empirical.predict(data['actions'], data['offsets_ns'], data['active'])
    zero = prediction.copy(); zero[..., :4] = (0., 0., 0., 1.)
    return dict(action_time=dict(indices=data['indices'], prediction=prediction),
        zero_motion_empirical_contact=dict(indices=data['indices'], prediction=zero)), missing


def verify_study_definition(d, study):
    verify_ordered_launch(d)
    for batch in BATCHES: verify_artifacts(output_root(batch), study.receipts[batch])


def execute(study, d, sequence_receipt):
    """Run one fixed comparison after preflight; stop on any infrastructure failure."""
    require(identity({k: d.get(k) for k in settings()}) == identity(settings()),
        'definition must bind actual experiment settings')
    create_output(OUTPUT); files = Artifacts(OUTPUT)
    active_fit = None; trainer = None; completed = []; ledger_path = None
    try:
        files.save('launch.json', dict(definition=d, definition_sha256=identity(d), sequence_receipt=sequence_receipt))
        report = study.report(); files.save('dataset.json', report)
        cov = coverage(study); files.save('coverage.json', cov)
        require(cov['eligible_to_fit'], 'predeclared coverage gate failed; retain exclusions, no fit')
        stream = AuditedStudyStream(study); view = stream.evaluation
        experiment_sha = files.hashes['launch.json']; dataset_sha = files.hashes['dataset.json']
        for seed in SEEDS:
            schedule = view.dataset.schedule('train', updates=UPDATES, batch_size=BATCH_SIZE, seed=seed)
            files.save(f'seed_{seed}_schedule.json', schedule)
            draws = [i for batch in schedule['batches'] for i in batch]
            empirical, exposure = view.fit_action_time(draws)
            files.save(f'seed_{seed}_baseline.json', dict(model=empirical.record(), exposure=exposure))
            heads = {}; missing = {}
            for role in ROLES: heads[role], missing[role] = baseline_heads(view, empirical, role)
            initial = None
            for variant in VARIANTS:
                for condition in CONDITIONS:
                    active_fit = f'seed_{seed}_{variant}_{condition}'
                    verify_study_definition(d, study)
                    trainer = CumulativePulseTrainer(condition, seed=seed, latent_dim=LATENT_DIM)
                    if initial is None: initial = trainer.initial_sha256
                    require(trainer.initial_sha256 == initial, 'same seeded initial model required across objectives/treatments')
                    binding = dict(experiment_sha256=experiment_sha, dataset_sha256=dataset_sha,
                        schedule_sha256=schedule['schedule_sha256'], input_variant=variant)
                    files.save(active_fit + '_request.json', dict(binding=binding, config=config(trainer),
                        planned_updates=UPDATES, initial_sha256=initial))
                    ledger_name = active_fit + '_updates.jsonl'; ledger_path = files.path(ledger_name)
                    ledger_hash = hashlib.sha256(); ledger_bytes = 0; count = 0; start = time.perf_counter()
                    with ledger_path.open('xb') as ledger:
                        def on_update(record):
                            nonlocal ledger_bytes, count
                            require(record['input_variant'] == binding['input_variant']
                                and record['schedule_sha256'] == binding['schedule_sha256']
                                and record['update'] == count + 1, 'exact treatment and optimizer accounting required')
                            raw = encode(record | dict(elapsed_seconds=time.perf_counter() - start))
                            require(ledger_bytes + len(raw) <= MAX_METADATA, 'bounded per-fit update ledger required')
                            files.capacity(len(raw)); ledger.write(raw); ledger.flush(); os.fsync(ledger.fileno())
                            files.used += len(raw); ledger_bytes += len(raw); ledger_hash.update(raw); count += 1
                            if count == 1 or count % 25 == 0:
                                print('MATCHED_STUDY_UPDATE', active_fit, count, flush=True)
                        fit = train_schedule(trainer, stream, schedule, on_update=on_update, input_variant=variant)
                    require(count == UPDATES and ledger_path.stat().st_size == ledger_bytes,
                        'complete exact update ledger required')
                    files.hashes[ledger_name] = ledger_hash.hexdigest()
                    verify_artifacts(OUTPUT, {ledger_name: files.hashes[ledger_name]})
                    snapshot, clone = files.snapshot(active_fit + '.pt', trainer, binding)
                    require(fit['input_variant'] == binding['input_variant'] == snapshot['binding']['input_variant'],
                        'fit/snapshot/input treatment identity mismatch')
                    files.save(active_fit + '_fit.json', dict(fit=fit, snapshot=snapshot,
                        elapsed_seconds=time.perf_counter() - start, ledger_sha256=files.hashes[ledger_name]))
                    for role in ROLES:
                        prediction = predict_heads(clone, stream, role=role, batch_size=INFERENCE_BATCH,
                            input_variant=binding['input_variant'])
                        require(prediction['input_variant'] == binding['input_variant']
                            and prediction['model_sha256'] == fit['model_sha256'] and prediction['updates'] == UPDATES,
                            'score only exact verified final snapshot and treatment')
                        files.save_predictions(active_fit + '_' + role + '.npz', prediction)
                        files.save(active_fit + '_' + role + '_prediction.json',
                            {k: v for k, v in prediction.items() if k != 'heads'})
                        for head, row in prediction['heads'].items():
                            heads[role][variant + '_' + condition + '_' + head] = row
                    verify_study_definition(d, study)
                    completed.append(active_fit); print('MATCHED_STUDY_FIT_COMPLETE', active_fit, flush=True)
                    active_fit = None; trainer = None; ledger_path = None
            for role in ROLES:
                files.save(f'seed_{seed}_{role}_scores.json', dict(
                    prediction=view.compare(heads[role], role=role),
                    matched_contact=evaluate_matched_hazards(view, study.prefixes, heads[role], role=role),
                    baseline_missing_cells=missing[role], seed=seed,
                    primary_heads={v + '_' + c: v + '_' + c + ('_direct_outcomes' if c == 'direct' else '_rollout_outcomes')
                        for v in VARIANTS for c in CONDITIONS},
                    no_best_seed_or_checkpoint_selection=True))
        # Reauthenticate original data after the complete fit sequence, not just
        # the policy leaves read during training/inference.
        reloaded = load_study(study.receipts)
        require(reloaded.report() == report and coverage(reloaded) == cov, 'study data changed during experiment')
        verify_study_definition(d, study); verify_artifacts(OUTPUT, files.hashes)
        require(len(completed) == len(SEEDS) * len(VARIANTS) * len(CONDITIONS), 'complete fixed fit roster required')
        files.save('result.json', dict(status='MATCHED_DEVELOPMENT_COMPARISON_COMPLETE',
            completed_fits=completed, fits=len(completed), optimizer_updates=len(completed) * UPDATES,
            seeds=list(SEEDS), objective_conditions=list(CONDITIONS), input_variants=list(VARIANTS),
            output_sha256=dict(files.hashes), artifact_bytes_before_result=files.used,
            final_evaluation=False, checkpoint_selection_performed=False,
            navigation_qualified=False, hardware_qualified=False, goal_achieved=False))
        print('MATCHED_STUDY_COMPLETE', len(completed), flush=True)
    except BaseException as error:
        # A partial final write/ledger is retained, never replaced or resumed.
        failure = dict(status='TERMINAL_MATCHED_STUDY_FAILURE', reason=repr(error),
            active_fit=active_fit, completed_fits=completed,
            actual_optimizer_updates_current_fit=trainer.updates if trainer is not None else None,
            partial_ledger=ledger_path.name if ledger_path is not None else None,
            output_sha256=dict(files.hashes), retry_performed=False, goal_achieved=False)
        write_json(OUTPUT / 'failure.json', failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence-result-sha256', required=True)
    parser.add_argument('--definition-sha256', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    study, d, receipt = preflight(args.sequence_result_sha256, args.definition_sha256)
    execute(study, d, receipt)


if __name__ == '__main__':
    main()
