"""Authenticate the distinct parallel result, then aggregate saved scores.

Read-only: no fitting, checkpoint loading, raw image/native data materialization,
or promotion. Requires explicit terminal and prospective definition digests.
Hash verification and saved-score aggregation are not an independent audit of
training or raw prediction scoring.
"""
import argparse
import json

from lewm.independent_pulse_scientific_readout_development import summarize
from scripts import run_go2_independent_pulse_matched_study_v1 as original
from scripts import run_go2_independent_pulse_parallel_study_v1 as parallel
from scripts.independent_pulse_parallel_fits_development import validate_job
from scripts.navigation_artifact_root_development import artifact_path, verify_artifacts


def read(name):
    path = artifact_path(parallel.OUTPUT, name)
    original.require(path.stat().st_size <= original.MAX_METADATA, 'bounded explicit parallel study metadata required')
    return json.loads(path.read_text())


def authenticate(result_sha256, definition_sha256):
    """Return authenticated terminal/launch/roster, without raw data or tensors."""
    output = parallel.OUTPUT
    verify_artifacts(output, {'result.json': result_sha256})
    original.require(not (output / 'failure.json').exists() and not (output / 'failure.json').is_symlink(),
        'failed parallel attempt cannot supply scientific results')
    terminal = read('result.json')
    roster = [f'seed_{s}_{v}_{c}' for s in original.SEEDS for v in original.VARIANTS for c in original.CONDITIONS]
    original.require(terminal['status'] == 'MATCHED_DEVELOPMENT_COMPARISON_COMPLETE'
        and terminal['execution_revision'] == 'parallel.v1' and terminal['completed_fits'] == roster
        and terminal['fits'] == 36 and terminal['optimizer_updates'] == 36 * original.UPDATES
        and terminal['seeds'] == list(original.SEEDS) and terminal['objective_conditions'] == list(original.CONDITIONS)
        and terminal['input_variants'] == list(original.VARIANTS) and terminal['maximum_parallel_fits'] == parallel.MAX_WORKERS
        and all(terminal[k] is False for k in ('final_evaluation', 'checkpoint_selection_performed',
            'navigation_qualified', 'hardware_qualified', 'goal_achieved')),
        'complete distinct unpromoted parallel 36-fit result required')
    expected = {'launch.json', 'hardware_preflight.json', 'dataset.json', 'coverage.json', 'fit_roster.json'} | {
        f'seed_{s}_{suffix}.json' for s in original.SEEDS for suffix in ('schedule', 'baseline')} | {
        f'seed_{s}_{r}_scores.json' for s in original.SEEDS for r in original.ROLES}
    for name in roster: expected.update(parallel.expected_fit_artifacts(name))
    bindings = terminal['output_sha256']
    original.require(set(bindings) == expected, 'complete exact parallel artifact roster required before access')
    verify_artifacts(output, bindings)
    launch = read('launch.json'); d = parallel.definition()
    original.require(original.identity(d) == original.identity(launch['definition'])
        == launch['definition_sha256'] == definition_sha256, 'unchanged prospective parallel definition required')
    jobs = read('fit_roster.json')
    original.require([j['name'] for j in jobs] == roster, 'exact canonical job roster required')
    schedules = {s: read(f'seed_{s}_schedule.json') for s in original.SEEDS}
    initial = {}
    for job in jobs:
        validate_job(job)
        original.require(job['seed'] in original.SEEDS and job['latent_dim'] == original.LATENT_DIM
            and job['schedule'] == schedules[job['seed']] and job['schedule']['updates'] == original.UPDATES
            and job['schedule']['batch_size'] == original.BATCH_SIZE
            and job['binding']['experiment_sha256'] == bindings['launch.json']
            and job['binding']['dataset_sha256'] == bindings['dataset.json'], 'original scientific job settings required')
        original.require(initial.setdefault(job['seed'], job['initial_sha256']) == job['initial_sha256'],
            'paired initialization required across every treatment/objective')
        name = job['name']; names = parallel.expected_fit_artifacts(name)
        complete = read(name + '_complete.json')
        receipt = dict(name=name, complete_sha256=bindings[name + '_complete.json'],
            output_sha256={n: bindings[n] for n in names}, worker_pid=complete['worker_pid'],
            artifact_bytes=sum(artifact_path(output, n).stat().st_size for n in names))
        parallel.admit_fit(output, job, receipt)
    original.require(sum(artifact_path(output, n).stat().st_size for n in bindings)
        == terminal['artifact_bytes_before_result'], 'whole study artifact accounting mismatch')
    sequence = launch['sequence_receipt']
    original.require(set(sequence) == {'launch.json', 'result.json'}
        and sequence['launch.json'] == original.SEQUENCE_LAUNCH, 'original collection receipt required')
    verify_artifacts(original.SEQUENCE, sequence)
    collection = original.read_json(original.SEQUENCE, 'result.json')
    original.require(collection['status'] == 'ALL12_FIXED_RGB_BODY_BATCH_RECEIPTS_VERIFIED'
        and collection['completed_batches'] == list(original.BATCHES)
        and collection['planned_layouts'] == 12 and collection['planned_episodes'] == 1440,
        'complete original twelve-layout collection required')
    verify_artifacts(original.SEQUENCE, collection['output_sha256'])
    verify_artifacts(output, {'result.json': result_sha256})
    original.require(original.identity(parallel.definition()) == definition_sha256,
        'parallel definition changed during terminal authentication')
    return terminal, launch


def read_result(result_sha256, definition_sha256):
    terminal, _ = authenticate(result_sha256, definition_sha256)
    scores = {s: {r: read(f'seed_{s}_{r}_scores.json') for r in original.ROLES} for s in original.SEEDS}
    report = summarize(scores)
    verify_artifacts(parallel.OUTPUT, terminal['output_sha256'])
    verify_artifacts(parallel.OUTPUT, {'result.json': result_sha256})
    original.require(original.identity(parallel.definition()) == definition_sha256,
        'parallel definition changed during scientific aggregation')
    report.update(authenticated_study_result_sha256=result_sha256,
        authenticated_study_definition_sha256=definition_sha256, execution_revision='parallel.v1',
        authenticated_score_sha256={f'seed_{s}_{r}_scores.json': terminal['output_sha256'][f'seed_{s}_{r}_scores.json']
            for s in original.SEEDS for r in original.ROLES}, study_output_hashes_verified=True,
        independent_training_or_raw_scoring_audit_performed=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-sha256', required=True)
    parser.add_argument('--definition-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(read_result(args.result_sha256, args.definition_sha256), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
