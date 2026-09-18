"""Actual owned spawn lifecycle; synthetic collection bytes, no native scene."""
from copy import deepcopy
from dataclasses import replace
import json
import multiprocessing
import os
import psutil
import pytest

from scripts import independent_round_trip_collection_process_development as lifecycle
from scripts import independent_round_trip_collection_handoff_development as handoff
from lewm.tests.test_independent_round_trip_population_case_evidence_development import saved
from lewm.tests.test_independent_round_trip_collection_handoff_development import setup, publish
from lewm.independent_round_trip_comparison_study_development import CASES
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest


def child_main(connection, release, code):
    connection.send(lifecycle.identity(psutil.Process()))
    connection.close()
    if not release.wait(30):
        raise RuntimeError('test parent did not release child')
    raise SystemExit(code)


@pytest.fixture
def bound(setup, monkeypatch):
    output = setup[0]
    launch = json.loads((output/'launch.json').read_text())
    launch['source_sha256'].update({n:digest(ROOT/n)
        for n in (lifecycle.SOURCE, lifecycle.TEST, lifecycle.PROTOCOL)})
    (output/'launch.json').write_text(json.dumps(launch)+'\n')
    def checked(root, expected, verifier):
        handoff.verify_artifacts(root, {'launch.json':expected})
        return deepcopy(launch)
    monkeypatch.setattr(handoff.runtime, 'checked_launch', checked)
    return (*setup[:3], digest(output/'launch.json'))


@pytest.fixture
def workers():
    children = []
    def start(code=0):
        context = multiprocessing.get_context('spawn')
        receive, send = context.Pipe(duplex=False)
        release = context.Event()
        process = context.Process(target=child_main, args=(send, release, code))
        process.start(); send.close()
        children.append((process, release))
        assert receive.poll(15), 'real spawn child failed to become ready'
        owner = receive.recv(); receive.close()
        return process, release, owner
    yield start
    for process, release in children:
        release.set(); process.join(10)
        if process.is_alive():
            process.terminate(); process.join(5)
        lifecycle._tickets.pop(process, None)
        process.close()


def register(bound, worker):
    return lifecycle.register(worker[0], bound[0], CASES[0], bound[3], None, None)


def synthetic_handoff(bound, owner):
    record, _ = publish(bound)
    # The fixture simulates the collection worker's data publication, while the
    # process handle, parent ownership, exit status and identity checks are real.
    record['collection_owner'] = owner
    path = bound[0]/(CASES[0].name+handoff.SUFFIX)
    path.write_text(json.dumps(record)+'\n')
    return digest(path)


def finish(worker):
    worker[1].set(); worker[0].join(10)
    assert not worker[0].is_alive()


def test_actual_owned_zero_exit_and_handoff_confirmed_once(bound, workers):
    worker = workers(); ticket = register(bound, worker)
    registration = json.loads((bound[0]/(CASES[0].name+lifecycle.REGISTRATION)).read_text())
    assert registration['collection_owner'] == worker[2]
    assert registration['parent_owner']['pid'] == os.getpid()
    sha = synthetic_handoff(bound, worker[2]); finish(worker)
    result = lifecycle.confirm(ticket, sha, None)
    assert result['owned_process_exitcode'] == 0
    assert result['parent_verified_zero_exit'] and result['complete_collection_bindings_verified']
    assert result['status'] == lifecycle.STATUS
    assert not any(result[k] for k in ('raw_audit_completed', 'audited_episode_complete',
        'native_scene_ownership_released', 'audit_execution_permitted', 'automatic_retry'))
    with pytest.raises(ValueError, match='unconsumed'): lifecycle.confirm(ticket, sha, None)


@pytest.mark.parametrize('code', [1, 17])
def test_real_nonzero_exit_never_confirmed(bound, workers, code):
    worker = workers(code); ticket = register(bound, worker)
    sha = synthetic_handoff(bound, worker[2]); finish(worker)
    assert worker[0].exitcode == code
    with pytest.raises(ValueError, match='code zero'): lifecycle.confirm(ticket, sha, None)
    assert not (bound[0]/(CASES[0].name+lifecycle.CONFIRMATION)).exists()


def test_live_child_rejected_before_reading_handoff(bound, workers, monkeypatch):
    worker = workers(); ticket = register(bound, worker)
    monkeypatch.setattr(handoff, 'read_for_audit', lambda *a:pytest.fail('live child admitted'))
    with pytest.raises(ValueError, match='code zero'): lifecycle.confirm(ticket, 'a'*64, None)


def test_already_ended_child_cannot_register(bound, workers):
    worker = workers(); finish(worker)
    with pytest.raises(ValueError, match='while live'): register(bound, worker)


def test_duplicate_and_forged_tickets_rejected(bound, workers):
    worker = workers(); ticket = register(bound, worker)
    with pytest.raises(ValueError, match='no retry'): register(bound, worker)
    with pytest.raises(ValueError, match='unconsumed'):
        lifecycle.confirm(replace(ticket), 'a'*64, None)
    replacement = workers()
    with pytest.raises(ValueError, match='unconsumed'):
        lifecycle.confirm(replace(ticket, process=replacement[0]), 'a'*64, None)


def test_nonprocess_handle_rejected(bound):
    with pytest.raises(ValueError, match='parent-owned'): register(bound, (object(),))


@pytest.mark.parametrize('fault', ['registration', 'handoff', 'collection', 'failure', 'launch'])
def test_changed_bound_bytes_or_failure_never_confirmed(bound, workers, fault):
    worker = workers(); ticket = register(bound, worker)
    sha = synthetic_handoff(bound, worker[2]); finish(worker)
    name = {'registration':CASES[0].name+lifecycle.REGISTRATION,
        'handoff':CASES[0].name+handoff.SUFFIX,
        'collection':CASES[0].name+'/command_tape.json',
        'failure':CASES[0].name+handoff.FAILURE_SUFFIX, 'launch':'launch.json'}[fault]
    (bound[0]/name).write_text('changed evidence')
    with pytest.raises(ValueError): lifecycle.confirm(ticket, sha, None)
    assert not (bound[0]/(CASES[0].name+lifecycle.CONFIRMATION)).exists()


def test_different_ended_collector_cannot_use_zero_exit_of_registered_child(bound, workers):
    worker = workers(); other = workers(); ticket = register(bound, worker)
    sha = synthetic_handoff(bound, other[2]); finish(worker); finish(other)
    with pytest.raises(ValueError, match='differs from parent registration: collection_owner'):
        lifecycle.confirm(ticket, sha, None)


def test_reused_birth_identity_is_not_an_ended_original(bound, workers, monkeypatch):
    worker = workers(); ticket = register(bound, worker)
    sha = synthetic_handoff(bound, worker[2]); finish(worker)
    def changed(owner): raise ValueError('original owner process identity changed')
    monkeypatch.setattr(handoff, 'owner_live', changed)
    with pytest.raises(ValueError, match='identity changed'): lifecycle.confirm(ticket, sha, None)


def test_sources_required_in_launch(bound, workers, monkeypatch):
    worker = workers()
    monkeypatch.setattr(handoff, 'checked_launch', lambda *a:{'source_sha256':{}})
    with pytest.raises(ValueError, match='must be frozen'): register(bound, worker)
