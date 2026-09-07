"""Four-worker scheduling revision of the unchanged 36-fit development study.

Never starts from a partial cohort or an existing sequential/parallel attempt.
The original runner remains unchanged and its scientific settings are inherited.
This revision requires its own prospective definition digest before execution.
"""
import argparse
from copy import deepcopy
import os
import json
import shutil
import sys
import time

import numpy as np
import torch

from scripts import run_go2_independent_pulse_matched_study_v1 as reference
from scripts.independent_pulse_parallel_fits_development import (
    MAX_WORKERS, FIT_BUDGET, PARENT_BUDGET, make_job, dispatch, fit_name)
from scripts.navigation_artifact_root_development import create_output, validate_root, verify_artifacts, artifact_path
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

SOURCE = 'scripts/run_go2_independent_pulse_parallel_study_v1.py'
TEST = 'lewm/tests/test_independent_pulse_parallel_study_development.py'
PROTOCOL = 'docs/go2_independent_pulse_parallel_study_v1_2026-09-07.md'
ORIGINAL_DEFINITION = '3a1b0d78ba9bb4cc1854bc4d1079a26f21ffa57652a7664aaff6af650318956b'
OUTPUT = reference.BASE / 'go2_independent_pulse_parallel_study_v1_attempt_001'
MIN_AVAILABLE_RAM = 32 * 1024**3


def hardware_observation():
    # Read-only OS capacity snapshot, not a benchmark or hard memory limit.
    with open('/proc/meminfo') as source:
        available = next(int(line.split()[1]) * 1024 for line in source if line.startswith('MemAvailable:'))
    return dict(cpu_affinity_count=len(os.sched_getaffinity(0)), available_ram_bytes=available,
        artifact_free_bytes=shutil.disk_usage(reference.BASE).free,
        workspace_free_bytes=shutil.disk_usage(reference.ROOT).free,
        workers=MAX_WORKERS, cpu_threads_per_worker=1, gpu_used=False,
        memory_limit_enforced=False, timestamp_unix_ns=time.time_ns())


def definition():
    old = reference.definition()
    reference.require(reference.identity(old) == ORIGINAL_DEFINITION, 'unchanged original scientific definition required')
    d = deepcopy(old)
    d.update(schema='independent_pulse_parallel_study.v1', output_root=str(OUTPUT),
        source_sha256=discover_sources((SOURCE, TEST, PROTOCOL), old['source_sha256']),
        original_definition_sha256=ORIGINAL_DEFINITION,
        process_start_method='spawn', maximum_parallel_fits=MAX_WORKERS,
        fit_artifact_allowance_bytes=FIT_BUDGET, parent_artifact_allowance_bytes=PARENT_BUDGET,
        minimum_available_ram_bytes=MIN_AVAILABLE_RAM, gpu_execution=False,
        canonical_result_order='seed,input_variant,objective',
        failure_behavior='stop dispatch, drain existing workers, retain failures; no retry')
    reference.verify_ordered_launch(d)
    return d


def preflight(sequence_result_sha256, definition_sha256):
    validate_root(OUTPUT, must_exist=False)
    reference.require(not OUTPUT.exists() and not OUTPUT.is_symlink(), 'exclusive parallel attempt; no retry/resume')
    d = definition()
    reference.require(reference.identity(d) == definition_sha256, 'reviewed exact parallel definition required')
    hw = hardware_observation()
    reference.require(hw['cpu_affinity_count'] >= MAX_WORKERS + 1
        and hw['available_ram_bytes'] >= MIN_AVAILABLE_RAM, 'parallel CPU/RAM capacity gate failed')
    # This retains all original interpreter/environment, completed-cohort,
    # source, storage and original-attempt exclusion checks. No monkeypatching.
    study, original, receipt = reference.preflight(sequence_result_sha256, ORIGINAL_DEFINITION)
    reference.require(reference.identity(original) == ORIGINAL_DEFINITION, 'original definition changed during preflight')
    reference.require(reference.identity(definition()) == definition_sha256, 'parallel definition changed during preflight')
    return study, d, receipt, hw


class ParentArtifacts(reference.Artifacts):
    def capacity(self, amount):
        reference.require(type(amount) is int and amount >= 0 and self.used + amount <= PARENT_BUDGET
            and shutil.disk_usage(self.output).free >= reference.RESERVE + amount,
            'parent artifact allowance or storage reserve exhausted')


def expected_fit_artifacts(name):
    return {name + x for x in ('_request.json', '_updates.jsonl', '.pt', '_fit.json', '_complete.json')} | {
        name + '_' + role + suffix for role in reference.ROLES for suffix in ('.npz', '_prediction.json')}


def admit_fit(output, job, receipt):
    """Authenticate the complete expected worker product before aggregation."""
    name = job['name']; bindings = receipt['output_sha256']
    reference.require(receipt['name'] == name and set(bindings) == expected_fit_artifacts(name)
        and receipt['complete_sha256'] == bindings[name + '_complete.json'], 'exact worker output roster required')
    reference.require(not (output / (name + '_failure.json')).exists()
        and not (output / (name + '_failure.json')).is_symlink(), 'failed worker cannot enter comparison')
    verify_artifacts(output, bindings)
    terminal = read_json(output, name + '_complete.json')
    reference.require(terminal['status'] == 'PARALLEL_MATCHED_FIT_COMPLETE' and terminal['name'] == name
        and terminal['job_sha256'] == reference.identity(job) and terminal['initial_sha256'] == job['initial_sha256']
        and terminal['updates'] == job['schedule']['updates'] and terminal['worker_pid'] == receipt['worker_pid']
        and terminal['cpu_threads'] == 1 and terminal['deterministic_algorithms'] is True
        and all(terminal[k] is False for k in ('retry_performed', 'navigation_qualified', 'goal_achieved'))
        and terminal['output_sha256'] == {k: v for k, v in bindings.items() if k != name + '_complete.json'},
        'complete bound deterministic worker required')
    total = sum(artifact_path(output, n).stat().st_size for n in bindings)
    reference.require(total == receipt['artifact_bytes'] and total <= FIT_BUDGET
        and total - artifact_path(output, name + '_complete.json').stat().st_size == terminal['artifact_bytes_before_terminal'],
        'worker artifact accounting mismatch')
    request = read_json(output, name + '_request.json'); fit = read_json(output, name + '_fit.json')
    expected_config = dict(condition=job['condition'], seed=job['seed'], latent_dim=job['latent_dim'],
        learning_rate=.001, ema_momentum=.99, updates=0)
    reference.require(request['binding'] == job['binding'] == fit['snapshot']['binding']
        and request['initial_sha256'] == fit['fit']['initial_sha256'] == job['initial_sha256']
        and request['planned_updates'] == fit['fit']['updates'] == terminal['updates']
        and fit['fit']['model_sha256'] == fit['snapshot']['model_sha256'] == terminal['model_sha256']
        and fit['fit']['input_variant'] == job['variant'] and fit['fit']['condition'] == job['condition']
        and fit['fit']['seed'] == job['seed']
        and fit['fit']['schedule_sha256'] == job['binding']['schedule_sha256']
        and fit['fit']['training_draw_indices'] == [i for b in job['schedule']['batches'] for i in b],
        'worker fit exposure/provenance mismatch')
    reference.require(request['config'] == expected_config
        and fit['snapshot']['configuration'] == expected_config | {'updates': terminal['updates']}
        and fit['snapshot']['filename'] == name + '.pt'
        and fit['snapshot']['sha256'] == bindings[name + '.pt']
        and fit['snapshot']['bytes'] == artifact_path(output, name + '.pt').stat().st_size
        and fit['snapshot']['evaluation_only_reload_verified'] is True
        and all(fit['snapshot'][k] is False for k in
            ('training_resume_authorized', 'checkpoint_selection_performed', 'navigation_qualified'))
        and fit['ledger_sha256'] == bindings[name + '_updates.jsonl'],
        'snapshot configuration or durable ledger binding mismatch')
    ledger = artifact_path(output, name + '_updates.jsonl')
    reference.require(ledger.stat().st_size <= reference.MAX_METADATA, 'bounded exact optimizer ledger required')
    count = 0; last = None
    with ledger.open() as source:
        for line in source:
            reference.require(count < job['schedule']['updates'], 'extra optimizer update in ledger')
            row = json.loads(line)
            reference.require(row['update'] == count + 1 and row['sample_indices'] == job['schedule']['batches'][count]
                and row['schedule_sha256'] == job['binding']['schedule_sha256']
                and row['input_variant'] == job['variant'], 'per-update draw/treatment accounting mismatch')
            count += 1; last = row
    reference.require(count == job['schedule']['updates'] and last['model_sha256'] == terminal['model_sha256'],
        'complete update ledger must end at the saved final model')
    return terminal


def collect_heads(output, view, seed, jobs, receipts):
    schedule = read_json(output, f'seed_{seed}_schedule.json')
    draws = [i for b in schedule['batches'] for i in b]
    empirical, exposure = view.fit_action_time(draws)
    baseline = read_json(output, f'seed_{seed}_baseline.json')
    reference.require(baseline == dict(model=empirical.record(), exposure=exposure), 'baseline exposure changed')
    heads = {}; missing = {}
    for role in reference.ROLES: heads[role], missing[role] = reference.baseline_heads(view, empirical, role)
    for job in jobs:
        if job['seed'] != seed: continue
        name = job['name']; terminal = admit_fit(output, job, receipts[name])
        expected = ('direct_outcomes',) if job['condition'] == 'direct' else ('direct_outcomes', 'rollout_outcomes')
        for role in reference.ROLES:
            meta = read_json(output, name + '_' + role + '_prediction.json')
            reference.require(meta['role'] == role and meta['condition'] == job['condition']
                and meta['seed'] == seed and meta['input_variant'] == job['variant']
                and meta['model_sha256'] == terminal['model_sha256'] and meta['updates'] == terminal['updates']
                and meta['primary_head'] == expected[-1], 'prediction metadata identity mismatch')
            with np.load(artifact_path(output, name + '_' + role + '.npz'), allow_pickle=False) as arrays:
                reference.require(set(arrays.files) == set(expected) | {'indices'}
                    and np.array_equal(arrays['indices'], view.arrays(role)['indices']), 'exact prediction rows/heads required')
                for head in expected:
                    heads[role][job['variant'] + '_' + job['condition'] + '_' + head] = dict(
                        indices=arrays['indices'].copy(), prediction=arrays[head].copy())
    return heads, missing


def execute(study, d, sequence_receipt, hw):
    reference.require(reference.identity({k: d.get(k) for k in reference.settings()}) == reference.identity(reference.settings())
        and d['maximum_parallel_fits'] == MAX_WORKERS and d['process_start_method'] == 'spawn',
        'unchanged scientific settings and fixed execution mode required')
    reference.require(not reference.OUTPUT.exists() and not reference.OUTPUT.is_symlink(),
        'original study must remain unlaunched; no duplicate comparison')
    create_output(OUTPUT); files = ParentArtifacts(OUTPUT); receipts = {}; jobs = []; started = time.perf_counter()
    try:
        files.save('launch.json', dict(definition=d, definition_sha256=reference.identity(d), sequence_receipt=sequence_receipt))
        files.save('hardware_preflight.json', hw)
        report = study.report(); files.save('dataset.json', report)
        cov = reference.coverage(study); files.save('coverage.json', cov)
        reference.require(cov['eligible_to_fit'], 'predeclared coverage gate failed; no fit')
        view = study.evaluation
        for seed in reference.SEEDS:
            schedule = view.dataset.schedule('train', updates=reference.UPDATES, batch_size=reference.BATCH_SIZE, seed=seed)
            files.save(f'seed_{seed}_schedule.json', schedule)
            empirical, exposure = view.fit_action_time([i for b in schedule['batches'] for i in b])
            files.save(f'seed_{seed}_baseline.json', dict(model=empirical.record(), exposure=exposure))
            initial = reference.CumulativePulseTrainer('direct', seed=seed, latent_dim=reference.LATENT_DIM).initial_sha256
            for variant in reference.VARIANTS:
                for condition in reference.CONDITIONS:
                    binding = dict(experiment_sha256=files.hashes['launch.json'], dataset_sha256=files.hashes['dataset.json'],
                        schedule_sha256=schedule['schedule_sha256'], input_variant=variant)
                    jobs.append(make_job(seed, variant, condition, schedule, binding, initial, latent_dim=reference.LATENT_DIM))
        files.save('fit_roster.json', jobs)
        def on_result(receipt):
            job = next(j for j in jobs if j['name'] == receipt['name'])
            admit_fit(OUTPUT, job, receipt)
            reference.require(receipt['name'] not in receipts, 'duplicate fit completion')
            receipts[receipt['name']] = receipt
            print('PARALLEL_MATCHED_FIT_COMPLETE', receipt['name'], len(receipts), len(jobs), flush=True)
        dispatch(jobs, initargs=(reference.AuditedStudyStream, study, OUTPUT,
            reference.verify_study_definition, (d, study)), on_result=on_result)
        for seed in reference.SEEDS:
            heads, missing = collect_heads(OUTPUT, view, seed, jobs, receipts)
            for role in reference.ROLES:
                files.save(f'seed_{seed}_{role}_scores.json', dict(prediction=view.compare(heads[role], role=role),
                    matched_contact=reference.evaluate_matched_hazards(view, study.prefixes, heads[role], role=role),
                    baseline_missing_cells=missing[role], seed=seed,
                    primary_heads={v + '_' + c: v + '_' + c + ('_direct_outcomes' if c == 'direct' else '_rollout_outcomes')
                        for v in reference.VARIANTS for c in reference.CONDITIONS}, no_best_seed_or_checkpoint_selection=True))
        reloaded = reference.load_study(study.receipts)
        reference.require(reloaded.report() == report and reference.coverage(reloaded) == cov, 'study data changed during comparison')
        reference.verify_study_definition(d, study)
        roster = [j['name'] for j in jobs]
        reference.require(set(receipts) == set(roster), 'complete factorial required')
        all_hashes = dict(files.hashes)
        for job in jobs:
            receipt = receipts[job['name']]; admit_fit(OUTPUT, job, receipt)
            reference.require(not set(all_hashes) & set(receipt['output_sha256']), 'disjoint artifact ownership required')
            all_hashes.update(receipt['output_sha256'])
        verify_artifacts(OUTPUT, all_hashes)
        total = files.used + sum(r['artifact_bytes'] for r in receipts.values())
        result = dict(status='MATCHED_DEVELOPMENT_COMPARISON_COMPLETE', execution_revision='parallel.v1',
            completed_fits=roster, fits=len(roster), optimizer_updates=len(roster) * reference.UPDATES,
            seeds=list(reference.SEEDS), objective_conditions=list(reference.CONDITIONS), input_variants=list(reference.VARIANTS),
            output_sha256=all_hashes, artifact_bytes_before_result=total, maximum_parallel_fits=MAX_WORKERS,
            elapsed_seconds=time.perf_counter() - started, final_evaluation=False, checkpoint_selection_performed=False,
            navigation_qualified=False, hardware_qualified=False, goal_achieved=False)
        reference.require(total + len(reference.encode(result)) <= reference.BUDGET, 'whole study artifact budget exceeded')
        files.save('result.json', result)
        print('PARALLEL_MATCHED_STUDY_COMPLETE', len(roster), flush=True)
    except BaseException as error:
        # dispatch has drained its live workers before reaching here. Retain
        # every scheduled job name even if a worker crashed before a receipt.
        reference.write_json(OUTPUT / 'failure.json', dict(status='TERMINAL_PARALLEL_MATCHED_STUDY_FAILURE',
            reason=repr(error)[:8192], accepted_fits=list(receipts), planned_fits=[j['name'] for j in jobs],
            accepted_worker_receipts=receipts, output_sha256=dict(files.hashes), retry_performed=False, goal_achieved=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence-result-sha256', required=True)
    parser.add_argument('--definition-sha256', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    execute(*preflight(args.sequence_result_sha256, args.definition_sha256))


if __name__ == '__main__':
    main()
