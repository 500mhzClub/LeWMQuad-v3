"""Actual synthetic artifact readout; no real study or raw data access."""
from copy import deepcopy
import hashlib
import json

import pytest

from lewm.tests.test_independent_pulse_parallel_study_development import small, SyntheticStudyStream, no_verify
from scripts import run_go2_independent_pulse_matched_study_v1 as original
from scripts import run_go2_independent_pulse_parallel_study_v1 as parallel
from scripts import independent_pulse_parallel_fits_development as fits
from scripts import read_go2_independent_pulse_parallel_science_v1 as reader
from scripts import navigation_artifact_root_development as authority


def store(root, name, value):
    # Exact generated synthetic test artifact only. No production replacement.
    raw = original.encode(value)
    (root / name).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def complete(small, monkeypatch):
    study, d = small
    monkeypatch.setattr(original, 'SEEDS', (2026091101, 2026091102, 2026091103))
    monkeypatch.setattr(original, 'UPDATES', 1)
    monkeypatch.setattr(original, 'SEQUENCE', parallel.OUTPUT.parent / 'go2_synthetic_sequence_attempt_001')
    authority.create_output(original.SEQUENCE)
    first = store(original.SEQUENCE, 'launch.json', {'synthetic': True})
    monkeypatch.setattr(original, 'SEQUENCE_LAUNCH', first)
    seqsha = store(original.SEQUENCE, 'result.json', dict(status='ALL12_FIXED_RGB_BODY_BATCH_RECEIPTS_VERIFIED',
        completed_batches=list(original.BATCHES), planned_layouts=12, planned_episodes=1440,
        output_sha256={'launch.json': first}))
    d = original.settings() | dict(maximum_parallel_fits=4, process_start_method='spawn')
    monkeypatch.setattr(parallel, 'definition', lambda: deepcopy(d))
    # Complete actual worker artifacts and scores, without repeating process
    # isolation tests. The main suite separately exercises real spawn dispatch.
    def dispatch(jobs, *, initargs, on_result):
        for job in jobs: on_result(fits.execute_fit(job, SyntheticStudyStream(study), parallel.OUTPUT, no_verify))
    monkeypatch.setattr(parallel, 'dispatch', dispatch)
    parallel.execute(study, d, {'launch.json': first, 'result.json': seqsha}, {'synthetic': True})
    path = parallel.OUTPUT / 'result.json'
    return hashlib.sha256(path.read_bytes()).hexdigest(), original.identity(d)


def test_complete_parallel_scientific_reader_preserves_scope_and_original_contrasts(complete):
    report = reader.read_result(*complete)
    assert report['execution_revision'] == 'parallel.v1'
    assert report['authenticated_study_result_sha256'] == complete[0]
    assert report['authenticated_study_definition_sha256'] == complete[1]
    assert len(report['authenticated_score_sha256']) == 9
    assert report['study_output_hashes_verified']
    assert not report['independent_training_or_raw_scoring_audit_performed']
    assert not report['goal_achieved'] and not report['navigation_qualified']


def reject_mutation(complete, fault):
    result_sha, definition_sha = complete
    if fault == 'terminal_sha': result_sha = '0' * 64
    elif fault == 'definition': definition_sha = '0' * 64
    elif fault == 'root_failure': store(parallel.OUTPUT, 'failure.json', {'synthetic': True})
    elif fault == 'fit_failure':
        store(parallel.OUTPUT, 'seed_2026091101_full_direct_failure.json', {'synthetic': True})
    elif fault == 'changed_score':
        store(parallel.OUTPUT, 'seed_2026091101_train_scores.json', {'synthetic': True})
    else:
        result = reader.read('result.json')
        if fault == 'partial_roster': result['completed_fits'].pop()
        elif fault == 'extra_artifact': result['output_sha256']['undeclared.json'] = '0' * 64
        elif fault == 'wrong_updates': result['optimizer_updates'] += 1
        elif fault == 'promotion': result['navigation_qualified'] = True
        else: result['execution_revision'] = 'sequential.v1'
        result_sha = store(parallel.OUTPUT, 'result.json', result)
    with pytest.raises(ValueError): reader.read_result(result_sha, definition_sha)


def test_bad_or_partial_parallel_artifacts_cannot_supply_readout(complete):
    # Reuse one complete synthetic factorial for ten isolated corruptions;
    # repeatedly training it adds no coverage to these read-only checks.
    names = ('result.json', 'failure.json', 'seed_2026091101_full_direct_failure.json',
        'seed_2026091101_train_scores.json')
    saved = {n: (parallel.OUTPUT / n).read_bytes() if (parallel.OUTPUT / n).exists() else None for n in names}
    for fault in ('terminal_sha', 'definition', 'root_failure', 'fit_failure',
            'partial_roster', 'extra_artifact', 'wrong_updates', 'promotion', 'original_identity', 'changed_score'):
        try:
            reject_mutation(complete, fault)
        finally:
            for name, raw in saved.items():
                path = parallel.OUTPUT / name
                if raw is not None: path.write_bytes(raw)
                elif path.exists(): path.unlink()
    reader.read_result(*complete)
