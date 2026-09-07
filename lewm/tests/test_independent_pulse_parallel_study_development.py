"""Real spawn isolation and reference parity on synthetic data only."""
from copy import deepcopy
import json
import os
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from lewm.tests.test_independent_pulse_matched_study_development import make_study
from lewm.tests.test_independent_pulse_study_runner_development import SyntheticStream
from scripts import independent_pulse_parallel_fits_development as fits
from scripts import run_go2_independent_pulse_matched_study_v1 as reference
from scripts import run_go2_independent_pulse_parallel_study_v1 as parallel
from scripts import navigation_artifact_root_development as authority
from scripts.cumulative_pulse_snapshot_development import equal_tree


class SyntheticStudyStream(SyntheticStream):
    def __init__(self, study):
        self.evaluation = study.evaluation; self.calls = []


def no_verify(*args):
    pass


def initialize_synthetic_worker(factory, study, output):
    authority.BASE = Path(output).parent
    fits.initialize_worker(factory, study, output, no_verify, ())


def synthetic_dispatch(jobs, *, initargs, on_result):
    factory, study, output, _, _ = initargs
    return fits.dispatch(jobs, initializer=initialize_synthetic_worker,
        initargs=(factory, study, output), on_result=on_result)


@pytest.fixture
def small(monkeypatch, tmp_path):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    monkeypatch.setattr(authority, 'BASE', tmp_path)
    monkeypatch.setattr(reference, 'OUTPUT', tmp_path / 'go2_synthetic_sequential_attempt_001')
    monkeypatch.setattr(parallel, 'OUTPUT', tmp_path / 'go2_synthetic_parallel_attempt_001')
    monkeypatch.setattr(reference, 'SEEDS', (37,))
    monkeypatch.setattr(reference, 'UPDATES', 2)
    monkeypatch.setattr(reference, 'BATCH_SIZE', 2)
    monkeypatch.setattr(reference, 'INFERENCE_BATCH', 2)
    monkeypatch.setattr(reference, 'LATENT_DIM', 8)
    study = make_study()
    monkeypatch.setattr(reference, 'AuditedStudyStream', SyntheticStudyStream)
    monkeypatch.setattr(reference, 'verify_study_definition', no_verify)
    monkeypatch.setattr(reference, 'load_study', lambda receipts: study)
    monkeypatch.setattr(parallel, 'dispatch', synthetic_dispatch)
    d = reference.settings() | dict(maximum_parallel_fits=4, process_start_method='spawn')
    return study, d


def read(root, name):
    return json.loads((root / name).read_text())


def jobs_for(study, *, variants=reference.VARIANTS, conditions=reference.CONDITIONS, updates=2, latent_dim=8):
    seed = 37
    schedule = study.evaluation.dataset.schedule('train', updates=updates, batch_size=2, seed=seed)
    initial = reference.CumulativePulseTrainer('direct', seed=seed, latent_dim=latent_dim).initial_sha256
    return [fits.make_job(seed, v, c, schedule, dict(experiment_sha256='a' * 64,
        dataset_sha256='b' * 64, schedule_sha256=schedule['schedule_sha256'], input_variant=v),
        initial, latent_dim=latent_dim) for v in variants for c in conditions]


def test_complete_spawned_factorial_is_numerically_identical_to_original_sequential_runner(small):
    study, definition = small
    parallel.execute(study, definition, {}, {'synthetic': True})
    reference.execute(study, reference.settings(), {})
    a, b = parallel.OUTPUT, reference.OUTPUT
    pa, pb = read(a, 'result.json'), read(b, 'result.json')
    assert pa['completed_fits'] == pb['completed_fits'] and pa['optimizer_updates'] == pb['optimizer_updates'] == 24
    assert pa['fits'] == 12 and pa['maximum_parallel_fits'] == 4 and not pa['goal_achieved']
    pids = set()
    for name in pa['completed_fits']:
        pids.add(read(a, name + '_complete.json')['worker_pid'])
        assert read(a, name + '_fit.json')['fit'] == read(b, name + '_fit.json')['fit']
        aa = read(a, name + '_request.json'); bb = read(b, name + '_request.json')
        assert aa['initial_sha256'] == bb['initial_sha256'] and aa['config'] == bb['config']
        def records(root):
            return [{k: v for k, v in json.loads(line).items() if k != 'elapsed_seconds'}
                for line in (root / (name + '_updates.jsonl')).read_text().splitlines()]
        assert records(a) == records(b)  # Every update loss, gradient, draw and model hash.
        for role in reference.ROLES:
            assert read(a, name + '_' + role + '_prediction.json') == read(b, name + '_' + role + '_prediction.json')
            with np.load(a / (name + '_' + role + '.npz'), allow_pickle=False) as x, \
                    np.load(b / (name + '_' + role + '.npz'), allow_pickle=False) as y:
                assert set(x.files) == set(y.files)
                for key in x.files: np.testing.assert_array_equal(x[key], y[key])
    # Short fits can finish before every spawned interpreter initializes.
    # The contract is bounded concurrent processes, not utilization by fiat.
    assert 2 <= len(pids) <= 4 and os.getpid() not in pids
    for role in reference.ROLES:
        assert read(a, 'seed_37_' + role + '_scores.json') == read(b, 'seed_37_' + role + '_scores.json')
    assert read(a, 'seed_37_schedule.json') == read(b, 'seed_37_schedule.json')
    assert read(a, 'seed_37_baseline.json') == read(b, 'seed_37_baseline.json')
    authority.verify_artifacts(a, pa['output_sha256']); authority.verify_artifacts(b, pb['output_sha256'])


@pytest.mark.parametrize('fault', ['name', 'binding', 'seed', 'initial', 'extra', 'updates'])
def test_invalid_requests_reject_before_dispatch(small, fault):
    study, _ = small; job = jobs_for(study)[0]
    if fault == 'name': job['name'] = 'sealed_test'
    elif fault == 'binding': job['binding']['input_variant'] = 'no_rgb'
    elif fault == 'seed': job['seed'] += 1
    elif fault == 'initial': job['initial_sha256'] = 'x' * 64
    elif fault == 'extra': job['extra'] = True
    else: job['schedule']['updates'] = 0
    with pytest.raises(ValueError): fits.dispatch([job], initargs=(), on_result=lambda r: None)
    assert not parallel.OUTPUT.exists()


@pytest.mark.parametrize('workers', [0, 5, True])
def test_worker_bound_is_explicit(small, workers):
    study, _ = small
    with pytest.raises(ValueError): fits.dispatch(jobs_for(study), initargs=(), on_result=lambda r: None, workers=workers)


def test_duplicate_roster_rejected_without_worker_or_output(small):
    study, _ = small; job = jobs_for(study)[0]
    with pytest.raises(ValueError, match='unique'): fits.dispatch([job, job], initargs=(), on_result=lambda r: None)
    assert not parallel.OUTPUT.exists()


def test_worker_failure_stops_dispatch_preserves_terminal_and_does_not_retry(small):
    study, _ = small; jobs = jobs_for(study, variants=('full',))
    jobs[0]['initial_sha256'] = '0' * 64
    authority.create_output(parallel.OUTPUT)
    with pytest.raises(ValueError, match='initial'):
        fits.dispatch(jobs, initializer=initialize_synthetic_worker,
            initargs=(SyntheticStudyStream, study, parallel.OUTPUT), on_result=lambda r: None, workers=1)
    failure = read(parallel.OUTPUT, jobs[0]['name'] + '_failure.json')
    assert failure['actual_optimizer_updates'] == 0 and not failure['retry_performed']
    assert not (parallel.OUTPUT / (jobs[1]['name'] + '_request.json')).exists()
    assert not (parallel.OUTPUT / 'result.json').exists()


def test_result_consumer_failure_does_not_start_later_jobs(small):
    study, _ = small; jobs = jobs_for(study, variants=('full',), conditions=('direct', 'jepa'))
    authority.create_output(parallel.OUTPUT)
    def reject(receipt): raise ValueError('synthetic parent receipt rejection')
    with pytest.raises(ValueError, match='receipt rejection'):
        fits.dispatch(jobs, initializer=initialize_synthetic_worker,
            initargs=(SyntheticStudyStream, study, parallel.OUTPUT), on_result=reject, workers=1)
    assert (parallel.OUTPUT / (jobs[0]['name'] + '_complete.json')).exists()
    assert not (parallel.OUTPUT / (jobs[1]['name'] + '_request.json')).exists()


@pytest.mark.parametrize('fault', ['training', 'snapshot', 'inference', 'source_after'])
def test_fit_failures_retain_actual_updates_and_partial_ledger(small, monkeypatch, fault):
    study, _ = small; job = jobs_for(study)[0]; authority.create_output(parallel.OUTPUT)
    stream = SyntheticStudyStream(study); calls = []
    def fail(*a, **kw): raise ValueError('synthetic failure')
    def verify():
        calls.append(True)
        if fault == 'source_after' and len(calls) == 2: fail()
    if fault == 'training': monkeypatch.setattr(stream, 'training_batch', fail)
    elif fault == 'snapshot': monkeypatch.setattr(reference, 'save_snapshot', fail)
    elif fault == 'inference': monkeypatch.setattr(reference, 'predict_heads', fail)
    with pytest.raises(ValueError, match='synthetic'): fits.execute_fit(job, stream, parallel.OUTPUT, verify)
    failure = read(parallel.OUTPUT, job['name'] + '_failure.json')
    assert failure['actual_optimizer_updates'] == (0 if fault == 'training' else 2)
    assert not failure['retry_performed'] and not (parallel.OUTPUT / (job['name'] + '_complete.json')).exists()
    assert (parallel.OUTPUT / failure['partial_ledger']).exists()
    before = (parallel.OUTPUT / (job['name'] + '_failure.json')).read_bytes()
    with pytest.raises(ValueError, match='exclusive'): fits.execute_fit(job, stream, parallel.OUTPUT, verify)
    assert (parallel.OUTPUT / (job['name'] + '_failure.json')).read_bytes() == before


def test_static_budgets_and_worker_filename_ownership(small, monkeypatch):
    assert 36 * fits.FIT_BUDGET + fits.PARENT_BUDGET < reference.BUDGET
    study, _ = small; job = jobs_for(study)[0]; authority.create_output(parallel.OUTPUT)
    files = fits.FitArtifacts(parallel.OUTPUT, job['name'])
    for name in ('result.json', 'seed_37_full_jepa_request.json', '../escape.json', 'sealed_test.json'):
        with pytest.raises(ValueError): files.save(name, {})
    monkeypatch.setattr(fits, 'FIT_BUDGET', fits.FAILURE_ALLOWANCE)
    with pytest.raises(ValueError, match='allowance'): files.save(job['name'] + '_request.json', {})
    files.failure({'synthetic': True})
    assert read(parallel.OUTPUT, job['name'] + '_failure.json') == {'synthetic': True}


@pytest.mark.parametrize('fault', ['roster', 'job', 'updates', 'bytes', 'failure'])
def test_corrupt_worker_receipts_cannot_enter_parent(small, fault):
    study, _ = small; job = jobs_for(study, variants=('full',), conditions=('direct',))[0]
    authority.create_output(parallel.OUTPUT)
    receipt = fits.execute_fit(job, SyntheticStudyStream(study), parallel.OUTPUT, no_verify)
    parallel.admit_fit(parallel.OUTPUT, job, receipt)
    if fault == 'roster': receipt['output_sha256'].pop(job['name'] + '.pt')
    elif fault == 'job': job['binding']['dataset_sha256'] = 'c' * 64
    elif fault == 'updates': job['schedule']['updates'] += 1
    elif fault == 'bytes': receipt['artifact_bytes'] += 1
    else: fits.FitArtifacts(parallel.OUTPUT, job['name']).failure({'synthetic': True})
    with pytest.raises(ValueError): parallel.admit_fit(parallel.OUTPUT, job, receipt)


@pytest.mark.parametrize('fault', ['draw', 'variant', 'extra', 'missing', 'final_model', 'snapshot_config'])
def test_rebound_invalid_update_accounting_is_independently_rejected(small, fault):
    study, _ = small; job = jobs_for(study, variants=('full',), conditions=('direct',))[0]
    authority.create_output(parallel.OUTPUT)
    receipt = fits.execute_fit(job, SyntheticStudyStream(study), parallel.OUTPUT, no_verify)
    name = job['name']; root = parallel.OUTPUT
    def store(filename, value):
        raw = reference.encode(value); (root / filename).write_bytes(raw)
        receipt['output_sha256'][filename] = __import__('hashlib').sha256(raw).hexdigest()
    fit = read(root, name + '_fit.json')
    if fault == 'snapshot_config': fit['snapshot']['configuration']['latent_dim'] += 1
    else:
        path = root / (name + '_updates.jsonl')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if fault == 'draw': rows[0]['sample_indices'][0] = 999
        elif fault == 'variant': rows[0]['input_variant'] = 'no_rgb'
        elif fault == 'extra': rows.append(deepcopy(rows[-1]))
        elif fault == 'missing': rows.pop()
        else: rows[-1]['model_sha256'] = '0' * 64
        raw = b''.join(reference.encode(r) for r in rows); path.write_bytes(raw)
        sha = __import__('hashlib').sha256(raw).hexdigest()
        receipt['output_sha256'][path.name] = fit['ledger_sha256'] = sha
    store(name + '_fit.json', fit)
    complete_name = name + '_complete.json'; terminal = read(root, complete_name)
    terminal['output_sha256'] = {k: v for k, v in receipt['output_sha256'].items() if k != complete_name}
    terminal['artifact_bytes_before_terminal'] = sum((root / n).stat().st_size for n in terminal['output_sha256'])
    store(complete_name, terminal)
    receipt['complete_sha256'] = receipt['output_sha256'][complete_name]
    receipt['artifact_bytes'] = sum((root / n).stat().st_size for n in receipt['output_sha256'])
    with pytest.raises(ValueError, match='accounting|ledger|configuration'):
        parallel.admit_fit(root, job, receipt)


def test_duplicate_original_attempt_rejected_without_parallel_output(small):
    study, d = small; authority.create_output(reference.OUTPUT)
    with pytest.raises(ValueError, match='duplicate'): parallel.execute(study, d, {}, {})
    assert not parallel.OUTPUT.exists()


def test_parent_failure_cannot_publish_partial_factorial(small, monkeypatch):
    study, d = small
    def broken(*a, **kw): raise RuntimeError('synthetic worker pool failure')
    monkeypatch.setattr(parallel, 'dispatch', broken)
    with pytest.raises(RuntimeError): parallel.execute(study, d, {}, {})
    failure = read(parallel.OUTPUT, 'failure.json')
    assert len(failure['planned_fits']) == 12 and failure['accepted_fits'] == []
    assert not failure['retry_performed'] and not (parallel.OUTPUT / 'result.json').exists()


def test_changed_science_rejected_before_output(small):
    study, d = small; d['updates_per_fit'] += 1
    with pytest.raises(ValueError, match='scientific'): parallel.execute(study, d, {}, {})
    assert not parallel.OUTPUT.exists()


@pytest.mark.parametrize('fault', ['definition', 'cpu', 'ram', 'original_gate', 'changed_source', 'existing'])
def test_parallel_entry_rejects_before_worker_or_output(small, monkeypatch, fault):
    study, d = small
    original_definition = reference.settings()
    monkeypatch.setattr(parallel, 'ORIGINAL_DEFINITION', reference.identity(original_definition))
    monkeypatch.setattr(parallel, 'definition', lambda: deepcopy(d))
    hardware = dict(cpu_affinity_count=32, available_ram_bytes=78 * 1024**3)
    if fault == 'cpu': hardware['cpu_affinity_count'] = 4
    elif fault == 'ram': hardware['available_ram_bytes'] = parallel.MIN_AVAILABLE_RAM - 1
    monkeypatch.setattr(parallel, 'hardware_observation', lambda: hardware)
    calls = []
    def original_gate(*args):
        calls.append(args)
        if fault == 'original_gate': raise ValueError('synthetic original cohort/environment/storage rejection')
        if fault == 'changed_source': d['changed'] = True
        return study, original_definition, {}
    monkeypatch.setattr(reference, 'preflight', original_gate)
    if fault == 'existing': authority.create_output(parallel.OUTPUT)
    with pytest.raises(ValueError): parallel.preflight('a' * 64,
        '0' * 64 if fault == 'definition' else reference.identity(d))
    assert parallel.OUTPUT.exists() == (fault == 'existing')
    if fault in ('definition', 'cpu', 'ram', 'existing'): assert not calls


def test_parallel_preflight_passes_exact_original_gate_and_records_capacity(small, monkeypatch):
    study, d = small; original_definition = reference.settings()
    monkeypatch.setattr(parallel, 'ORIGINAL_DEFINITION', reference.identity(original_definition))
    monkeypatch.setattr(parallel, 'definition', lambda: deepcopy(d))
    hardware = dict(cpu_affinity_count=32, available_ram_bytes=78 * 1024**3)
    monkeypatch.setattr(parallel, 'hardware_observation', lambda: hardware)
    calls = []
    def original_gate(*args):
        calls.append(args); return study, original_definition, {'synthetic': True}
    monkeypatch.setattr(reference, 'preflight', original_gate)
    result = parallel.preflight('a' * 64, reference.identity(d))
    assert result == (study, d, {'synthetic': True}, hardware)
    assert calls == [('a' * 64, reference.identity(original_definition))]
    assert not parallel.OUTPUT.exists()


def test_production_width_batch_all_seeds_and_arms_spawn_parity(small, monkeypatch):
    study, d = small
    monkeypatch.setattr(reference, 'SEEDS', (2026091101, 2026091102, 2026091103))
    monkeypatch.setattr(reference, 'UPDATES', 6)
    monkeypatch.setattr(reference, 'BATCH_SIZE', 6)
    monkeypatch.setattr(reference, 'INFERENCE_BATCH', 6)
    monkeypatch.setattr(reference, 'LATENT_DIM', 32)
    d = reference.settings() | dict(maximum_parallel_fits=4, process_start_method='spawn')
    parallel.execute(study, d, {}, {'synthetic': True})
    reference.execute(study, reference.settings(), {})
    a, b = parallel.OUTPUT, reference.OUTPUT
    result = read(a, 'result.json')
    assert result['fits'] == 36 and result['optimizer_updates'] == 216
    for name in result['completed_fits']:
        assert read(a, name + '_fit.json')['fit'] == read(b, name + '_fit.json')['fit']
        for role in reference.ROLES:
            with np.load(a / (name + '_' + role + '.npz'), allow_pickle=False) as x, \
                    np.load(b / (name + '_' + role + '.npz'), allow_pickle=False) as y:
                for key in x.files: np.testing.assert_array_equal(x[key], y[key])
    for seed in reference.SEEDS:
        for role in reference.ROLES:
            assert read(a, f'seed_{seed}_{role}_scores.json') == read(b, f'seed_{seed}_{role}_scores.json')


def test_longer_schedule_spawn_parity_and_measured_resources(small, record_property):
    study, _ = small
    seed = 2026091101
    schedule = study.evaluation.dataset.schedule('train', updates=1200, batch_size=6, seed=seed)
    initial = reference.CumulativePulseTrainer('direct', seed=seed, latent_dim=32).initial_sha256
    jobs = [fits.make_job(seed, v, c, schedule, dict(experiment_sha256='a' * 64,
        dataset_sha256='b' * 64, schedule_sha256=schedule['schedule_sha256'], input_variant=v),
        initial, latent_dim=32) for v, c in (('full', 'direct'), ('full', 'supervised_rollout'),
            ('full', 'jepa'), ('no_rgb', 'jepa'))]
    authority.create_output(reference.OUTPUT); authority.create_output(parallel.OUTPUT)
    start = time.perf_counter()
    for job in jobs: fits.execute_fit(job, SyntheticStudyStream(study), reference.OUTPUT, no_verify)
    serial_seconds = time.perf_counter() - start
    start = time.perf_counter()
    receipts = fits.dispatch(jobs, initializer=initialize_synthetic_worker,
        initargs=(SyntheticStudyStream, study, parallel.OUTPUT), on_result=lambda r: None)
    parallel_seconds = time.perf_counter() - start
    pids = set(); cpu = []; peaks = []
    for job, receipt in zip(jobs, receipts, strict=True):
        name = job['name']; terminal = parallel.admit_fit(parallel.OUTPUT, job, receipt)
        pids.add(terminal['worker_pid']); cpu.append(terminal['worker_cpu_seconds'])
        peaks.append(terminal['process_lifetime_peak_rss_bytes'])
        assert terminal['updates'] == 1200 and terminal['cpu_threads'] == 1
        def records(root):
            return [{k: v for k, v in json.loads(line).items() if k != 'elapsed_seconds'}
                for line in (root / (name + '_updates.jsonl')).read_text().splitlines()]
        assert records(reference.OUTPUT) == records(parallel.OUTPUT)
        with (reference.OUTPUT / (name + '.pt')).open('rb') as a, (parallel.OUTPUT / (name + '.pt')).open('rb') as b:
            assert equal_tree(torch.load(a, map_location='cpu', weights_only=True),
                torch.load(b, map_location='cpu', weights_only=True))
        for role in reference.ROLES:
            with np.load(reference.OUTPUT / (name + '_' + role + '.npz'), allow_pickle=False) as a, \
                    np.load(parallel.OUTPUT / (name + '_' + role + '.npz'), allow_pickle=False) as b:
                for key in a.files: np.testing.assert_array_equal(a[key], b[key])
    assert 2 <= len(pids) <= 4 and os.getpid() not in pids
    assert all(c > 0 for c in cpu) and all(p > 0 for p in peaks)
    measured = dict(synthetic_only=True, jobs=4, updates_per_job=1200, batch_size=6, latent_dim=32,
        serial_seconds=serial_seconds, parallel_seconds=parallel_seconds,
        serial_over_parallel_wall_ratio=serial_seconds / parallel_seconds,
        worker_cpu_seconds_sum=sum(cpu), worker_cpu_over_parallel_wall=sum(cpu) / parallel_seconds,
        worker_process_lifetime_peak_rss_bytes=peaks,
        distinct_workers=len(pids), includes_spawn_and_artifact_io=True,
        actual_cohort_materialization_exercised=False, real_study_speedup_established=False)
    record_property('parallel_resource_observation', json.dumps(measured, sort_keys=True))
    print('SYNTHETIC_PARALLEL_RESOURCE_OBSERVATION', json.dumps(measured, sort_keys=True), flush=True)
