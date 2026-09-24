"""Preserved terminal failures, unchanged native science and waiting semantics."""
from copy import deepcopy

import pytest

from scripts import native_waiter_dispatch_abort_development as aborted
from scripts import run_go2_measured_plane_dispatch_recovery_v1 as candidate


@pytest.mark.parametrize('case', aborted.CASES)
def test_exact_terminal_scheduler_failure_is_admitted_without_success_label(case):
    name, pid, created, _, _, _, status, reason = case
    owner = aborted.owner(name, pid, created)
    launch = dict(boot_id=aborted.BOOT, waiter_pid=pid)
    failure = dict(status=status, reason=reason, automatic_retry=False)
    aborted.require_no_dispatch(launch, failure, [{'status': 'WAITING'}], owner, status, reason)
    for changed in (failure | {'status': 'COMPLETE'}, failure | {'reason': 'different'},
                    failure | {'automatic_retry': True}):
        with pytest.raises(ValueError):
            aborted.require_no_dispatch(launch, changed, [], owner, status, reason)
    with pytest.raises(ValueError):
        aborted.require_no_dispatch(launch, failure, [{'status': 'NATIVE_CHILD_STARTED'}], owner, status, reason)
    with pytest.raises(ValueError):
        aborted.require_no_dispatch(launch | {'boot_id': 'different'}, failure, [], owner, status, reason)


def test_scientific_definition_changes_only_output_root():
    old = candidate.original.definition()
    new = candidate.definition()
    assert new['output_root'] == str(candidate.OUTPUT)
    assert old['output_root'] != new['output_root']
    new['output_root'] = old['output_root']
    assert new == old


def test_worker_and_prefix_keep_original_code_and_all_other_global_bindings():
    for function, original, changes in (
        (candidate._worker, candidate.original.worker,
         {'OUTPUT', 'PROTOCOL', 'verify_inputs', 'prefix_result'}),
        (candidate.prefix_result, candidate.original.prefix_result, {'OUTPUT'}),
        (candidate.definition, candidate.original.definition, {'OUTPUT'})):
        assert function.__code__ is original.__code__
        assert function.__defaults__ == original.__defaults__
        assert function.__kwdefaults__ == original.__kwdefaults__
        assert set(function.__globals__) == set(original.__globals__)
        assert all(function.__globals__[key] is value for key, value in original.__globals__.items()
            if key not in changes)
    assert candidate._worker.__globals__['verify_inputs'] is candidate.verify_inputs
    assert candidate._worker.__globals__['pipeline'] is candidate.original.pipeline
    assert candidate._worker.__globals__['require_worker'] is candidate.original.require_worker
    assert candidate.prefix_result.__globals__['prefix'] is candidate.original.prefix


def test_existing_competitor_causes_wait_before_dispatch(monkeypatch):
    observations = iter([[{'pid': 42, 'command': ['run_go2_read_only_replay.py']}], [], []])
    slept = []
    monkeypatch.setattr(candidate, 'competitors', lambda: next(observations))
    monkeypatch.setattr(candidate.time, 'sleep', slept.append)
    candidate.wait_for_idle()
    assert slept == [30]


def test_transient_process_observation_error_cannot_start_scene(monkeypatch):
    def interrupted(): raise PermissionError('process observation interrupted')
    monkeypatch.setattr(candidate, 'competitors', interrupted)
    with pytest.raises(PermissionError): candidate.wait_for_idle()


def test_spawn_wrapper_dispatches_only_exact_private_worker(monkeypatch):
    calls = []
    def worker(sha):
        calls.append(sha)
        return {'receipt': sha}
    monkeypatch.setattr(candidate, '_worker', worker)
    assert candidate.worker('bound-launch') == {'receipt': 'bound-launch'}
    assert calls == ['bound-launch']


def test_prefix_negative_outcome_is_preserved():
    result = candidate.prefix_result({'decisions': 12}, {})
    assert result['status'] == 'PROSPECTIVE_BOUNDARY_NOT_REACHED'
    assert result['full_raw_audit_retained']
    assert not result['actual_paired_execution_compared']
    assert not result['navigation_verified']
