"""Real parent-owned audit process lifetime; synthetic collection and raw audit."""
from dataclasses import replace
import json
import pytest

from scripts import independent_round_trip_audit_process_development as parent
from scripts import independent_round_trip_separate_audit_development as separate
from scripts import independent_round_trip_collection_process_development as lifecycle
from scripts import independent_round_trip_collection_handoff_development as handoff
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest
from lewm.tests.test_independent_round_trip_population_case_evidence_development import saved
from lewm.tests.test_independent_round_trip_collection_handoff_development import setup
from lewm.tests.test_independent_round_trip_separate_audit_development import fixture as base_fixture, worker, run
from lewm.tests.test_independent_round_trip_collection_process_development import workers, finish


@pytest.fixture
def fixture(base_fixture, monkeypatch):
    f = base_fixture; output = f['output']; case = f['case']
    monkeypatch.setattr(parent, '_tickets', {})
    f['launch']['source_sha256'].update({n:digest(ROOT/n) for n in (parent.SOURCE, parent.TEST, parent.PROTOCOL)})
    (output/'launch.json').write_text(json.dumps(f['launch'])+'\n')
    f['launch_sha'] = digest(output/'launch.json')
    for suffix in (handoff.SUFFIX, lifecycle.REGISTRATION, lifecycle.CONFIRMATION):
        path = output/(case.name+suffix); record = json.loads(path.read_text())
        record['source_sha256'] = f['launch']['source_sha256']; record['launch_sha256'] = f['launch_sha']
        if 'artifact_sha256' in record:
            record['artifact_sha256']['launch.json'] = f['launch_sha']
        if suffix == lifecycle.CONFIRMATION:
            for bound_suffix in (handoff.SUFFIX, lifecycle.REGISTRATION):
                name = case.name+bound_suffix; record['artifact_sha256'][name] = digest(output/name)
            f['confirmation'] = record
        path.write_text(json.dumps(record)+'\n')
    f['confirmation_sha'] = digest(output/(case.name+lifecycle.CONFIRMATION))
    return f


def register(f, child):
    return parent.register(child[0], f['output'], f['case'], f['confirmation_sha'], f['launch_sha'], None, None)


def run_for_child(f, child, monkeypatch):
    monkeypatch.setattr(separate, 'require_audit_child', lambda confirmation:child[2])
    return run(f)


def test_zero_exit_with_complete_audit_accepted_once(fixture, worker, workers, monkeypatch):
    child = workers(); ticket = register(fixture, child)
    returned = run_for_child(fixture, child, monkeypatch); finish(child)
    result = parent.accept(ticket, returned, None)
    assert result['population_case_accepted'] and result['parent_verified_audit_zero_exit']
    assert result['owned_audit_exitcode'] == 0
    assert result['audit_owner'] == child[2]
    assert not result['verified_round_trip'] and not result['scientific_success_required']
    assert not result['same_process_collection_and_raw_audit']
    assert not result['native_scene_ownership_released'] and not result['goal_achieved']
    assert fixture['case'].name+handoff.LOG_SUFFIX in result['artifact_sha256']
    assert (fixture['output']/(fixture['case'].name+parent.COMPLETION)).exists()
    with pytest.raises(ValueError, match='unconsumed'): parent.accept(ticket, returned, None)


def test_real_nonzero_exit_rejects_even_complete_audit(fixture, worker, workers, monkeypatch):
    child = workers(23); ticket = register(fixture, child)
    returned = run_for_child(fixture, child, monkeypatch); finish(child)
    assert child[0].exitcode == 23
    with pytest.raises(ValueError, match='code zero'): parent.accept(ticket, returned, None)
    assert not (fixture['output']/(fixture['case'].name+parent.COMPLETION)).exists()


def test_live_child_cannot_be_accepted_before_evidence_reads(fixture, workers, monkeypatch):
    child = workers(); ticket = register(fixture, child)
    monkeypatch.setattr(separate, 'read_completed_audit', lambda *a:pytest.fail('live child admitted'))
    with pytest.raises(ValueError, match='code zero'): parent.accept(ticket, {}, None)


def test_ended_child_cannot_register(fixture, workers):
    child = workers(); finish(child)
    with pytest.raises(ValueError, match='while live'): register(fixture, child)


def test_forged_and_replaced_tickets_cannot_resume(fixture, workers):
    child = workers(); ticket = register(fixture, child)
    with pytest.raises(ValueError, match='no retry'): register(fixture, child)
    with pytest.raises(ValueError, match='unconsumed'): parent.accept(replace(ticket), {}, None)
    other = workers()
    with pytest.raises(ValueError, match='unconsumed'):
        parent.accept(replace(ticket, process=other[0]), {}, None)


def test_another_ended_auditor_cannot_borrow_registered_zero_exit(fixture, worker, workers, monkeypatch):
    child = workers(); other = workers(); ticket = register(fixture, child)
    returned = run_for_child(fixture, other, monkeypatch); finish(child); finish(other)
    with pytest.raises(ValueError, match='differs from original child registration: audit_owner'):
        parent.accept(ticket, returned, None)


@pytest.mark.parametrize('target', ['registration', 'confirmation', 'collection', 'audit_log', 'failure'])
def test_post_exit_changed_evidence_or_failure_cannot_be_accepted(fixture, worker, workers, monkeypatch, target):
    child = workers(); ticket = register(fixture, child)
    returned = run_for_child(fixture, child, monkeypatch); finish(child)
    case = fixture['case']
    name = {'registration':case.name+parent.REGISTRATION,
        'confirmation':case.name+lifecycle.CONFIRMATION, 'collection':case.name+'/command_tape.json',
        'audit_log':case.name+'_worker.log', 'failure':case.name+'_worker_failure.json'}[target]
    (fixture['output']/name).write_text('changed original evidence')
    with pytest.raises(ValueError): parent.accept(ticket, returned, None)
    assert not (fixture['output']/(case.name+parent.COMPLETION)).exists()


def test_failed_raw_audit_is_not_accepted_even_with_zero_exit(fixture, worker, workers, monkeypatch):
    child = workers(); ticket = register(fixture, child)
    def failed(*a, **k): raise RuntimeError('synthetic raw audit failure')
    monkeypatch.setattr(separate, 'audit', failed)
    returned = run_for_child(fixture, child, monkeypatch); finish(child)
    assert child[0].exitcode == 0 and returned['status'] == separate.FAILED
    with pytest.raises(ValueError, match='audit failure preserved'): parent.accept(ticket, returned, None)


def test_started_audit_cannot_be_registered_late(fixture, worker, workers, monkeypatch):
    child = workers(); run_for_child(fixture, child, monkeypatch)
    with pytest.raises(ValueError, match='fresh exclusive audit'): register(fixture, child)
