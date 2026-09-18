"""Synthetic split audit integration; no native scene or trained-model call."""
from copy import deepcopy
import json
import psutil
import pytest

from scripts import independent_round_trip_separate_audit_development as separate
from scripts import independent_round_trip_collection_process_development as lifecycle
from scripts import independent_round_trip_collection_handoff_development as handoff
from scripts import independent_round_trip_population_case_evidence_development as evidence
from scripts import independent_round_trip_population_runtime_development as runtime
from lewm.tests.test_independent_round_trip_population_case_evidence_development import saved
from lewm.tests.test_independent_round_trip_collection_handoff_development import setup, publish
from lewm.independent_round_trip_comparison_study_development import CASES
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest


@pytest.fixture
def fixture(setup, monkeypatch):
    output, collections, reports, _ = setup; case = CASES[0]
    launch = dict(synthetic_fixture=True, protocol=separate.PROTOCOL,
        input_admission={'factory_correction_admission':{'synthetic':True}},
        source_sha256={n:digest(ROOT/n) for n in (handoff.SOURCE, handoff.TEST, handoff.PROTOCOL,
            lifecycle.SOURCE, lifecycle.TEST, lifecycle.PROTOCOL,
            separate.SOURCE, separate.TEST, separate.PROTOCOL)})
    (output/'launch.json').write_text(json.dumps(launch)+'\n')
    launch_sha = digest(output/'launch.json')
    def checked(root, expected, verifier):
        handoff.verify_artifacts(root, {'launch.json':expected})
        return deepcopy(launch)
    monkeypatch.setattr(handoff.runtime, 'checked_launch', checked)
    record, _ = publish((output, collections, reports, launch_sha))
    # Synthetic receipts stand in for the real lifecycle already exercised in
    # test_independent_round_trip_collection_process_development.py.
    record['collection_owner'] = dict(pid=99999999, created=1.0, command=['synthetic-collector'])
    handoff_name = case.name+handoff.SUFFIX
    (output/handoff_name).write_text(json.dumps(record)+'\n')
    parent = lifecycle.identity(psutil.Process())
    registered = {k:record[k] for k in ('case', 'boot_id', 'launch_sha256',
        'reference_worker_sha256', 'source_sha256', 'collection_owner')}
    registered.update(status='INDEPENDENT_POPULATION_COLLECTION_PROCESS_REGISTERED',
        parent_owner=parent, start_method='spawn', automatic_retry=False,
        parent_verified_zero_exit=False, raw_audit_completed=False, audited_episode_complete=False,
        native_scene_ownership_released=False, audit_execution_permitted=False)
    reg_name = case.name+lifecycle.REGISTRATION
    (output/reg_name).write_text(json.dumps(registered)+'\n')
    confirmation = deepcopy(registered)
    confirmation.update(status=lifecycle.STATUS, original_collector_ended=True,
        owned_process_exitcode=0, parent_verified_zero_exit=True, complete_collection_bindings_verified=True,
        artifact_sha256={reg_name:digest(output/reg_name), handoff_name:digest(output/handoff_name),
            'launch.json':launch_sha})
    conf_name = case.name+lifecycle.CONFIRMATION
    (output/conf_name).write_text(json.dumps(confirmation)+'\n')
    (output/(case.name+'_worker.log')).unlink()  # Remove only the synthetic placeholder.
    return dict(output=output, case=case, confirmation=confirmation,
        confirmation_sha=digest(output/conf_name), launch=launch, launch_sha=launch_sha,
        report=reports[0], collection=collections[0])


def admit(fixture):
    f = fixture
    return separate.admit_collection(f['output'], f['case'], f['confirmation_sha'], f['launch_sha'], None, None)


def run(fixture):
    f = fixture
    return separate.audit_worker(f['output'], f['case'], f['confirmation_sha'], f['launch_sha'], None, None)


def read(fixture, returned):
    f = fixture
    return separate.read_completed_audit(f['output'], f['case'], returned,
        f['confirmation_sha'], f['launch_sha'], None, None)


@pytest.fixture
def worker(fixture, monkeypatch):
    events = []
    def geometry(path):
        assert path == separate.URDF
        events.append('geometry'); return 'synthetic geometry'
    def audit(case, collection, definition, *, input_root, robot_geometry, correction_admission):
        events.append('original_audit')
        assert case == fixture['case'] and collection == fixture['collection']
        assert input_root == fixture['output'] and robot_geometry == 'synthetic geometry'
        assert definition == fixture['launch']['source_sha256'][separate.PROTOCOL]
        assert correction_admission == {'synthetic':True}
        print('synthetic original audit')
        return deepcopy(fixture['report'])
    monkeypatch.setattr(separate, 'ArticulatedCollisionGeometry', geometry)
    monkeypatch.setattr(separate, 'audit', audit)
    monkeypatch.setattr(separate, 'require_audit_child', lambda confirmation:
        dict(pid=99999998, created=2.0, command=['synthetic-auditor']))
    monkeypatch.setattr(runtime, 'collect', lambda *a, **k:pytest.fail('collection is forbidden in audit worker'))
    return events


def test_confirmation_and_full_collection_admitted(fixture):
    confirmed, original = admit(fixture)
    assert confirmed == fixture['confirmation']
    assert original['collection'] == fixture['collection']


def test_actual_parent_cannot_invoke_audit_as_its_own_child(fixture):
    with pytest.raises(ValueError, match='fresh audit spawn child'):
        separate.require_audit_child(fixture['confirmation'])


def test_worker_persists_distinct_execution_and_reader_leaves_parent_acceptance_pending(fixture, worker):
    result = run(fixture)
    assert result['status'] == separate.WORKER_STATUS
    assert worker == ['geometry', 'original_audit']
    admitted = read(fixture, result)
    execution = admitted['execution']
    assert admitted['saved_audit_evidence_authenticated']
    assert not admitted['population_case_accepted'] and not admitted['parent_verified_audit_zero_exit']
    assert not execution['same_process_collection_and_raw_audit']
    assert not execution['native_collection_called'] and not execution['cpu_only_execution_qualified']
    assert not (fixture['output']/(fixture['case'].name+'_worker_execution.json')).exists()
    assert fixture['case'].name+handoff.LOG_SUFFIX in admitted['artifact_sha256']
    assert 'synthetic original audit' in (fixture['output']/(fixture['case'].name+'_worker.log')).read_text()


@pytest.mark.parametrize('field,value', [('owned_process_exitcode', True),
    ('owned_process_exitcode', 2), ('parent_verified_zero_exit', False),
    ('raw_audit_completed', True), ('native_scene_ownership_released', True),
    ('boot_id', 'changed'), ('reference_worker_sha256', 'a'*64)])
def test_rebound_invalid_confirmation_rejected(fixture, field, value):
    c = deepcopy(fixture['confirmation']); c[field] = value
    path = fixture['output']/(fixture['case'].name+lifecycle.CONFIRMATION)
    path.write_text(json.dumps(c)+'\n'); fixture['confirmation_sha'] = digest(path)
    with pytest.raises(ValueError): admit(fixture)


@pytest.mark.parametrize('target', ['registration', 'collection', 'collection_log', 'handoff'])
def test_original_bound_artifact_tamper_rejected(fixture, target):
    case = fixture['case']
    name = {'registration':case.name+lifecycle.REGISTRATION,
        'collection':case.name+'/command_tape.json', 'collection_log':case.name+handoff.LOG_SUFFIX,
        'handoff':case.name+handoff.SUFFIX}[target]
    (fixture['output']/name).write_text('changed original bytes')
    with pytest.raises(ValueError): admit(fixture)


def test_registration_parent_must_match_confirmation_even_when_rehashed(fixture):
    c = deepcopy(fixture['confirmation']); name = fixture['case'].name+lifecycle.REGISTRATION
    path = fixture['output']/name; registered = json.loads(path.read_text())
    registered['parent_owner']['created'] += 1
    path.write_text(json.dumps(registered)+'\n'); c['artifact_sha256'][name] = digest(path)
    path = fixture['output']/(fixture['case'].name+lifecycle.CONFIRMATION)
    path.write_text(json.dumps(c)+'\n'); fixture['confirmation_sha'] = digest(path)
    with pytest.raises(ValueError, match='disagree: parent_owner'): admit(fixture)


@pytest.mark.parametrize('fault', ['audit', 'raw_changed', 'persistence'])
def test_failed_audit_preserves_evidence_and_cannot_be_retried(fixture, worker, monkeypatch, fault):
    if fault == 'persistence':
        def bad(*a, **k): raise RuntimeError('synthetic persistence failure')
        monkeypatch.setattr(evidence, 'persist_audited_case', bad)
    else:
        def bad(*a, **k):
            if fault == 'audit': raise RuntimeError('synthetic original audit failure')
            (fixture['output']/(fixture['case'].name+'/command_tape.json')).write_text('changed during audit')
            return deepcopy(fixture['report'])
        monkeypatch.setattr(separate, 'audit', bad)
    returned = run(fixture)
    assert returned['status'] == separate.FAILED
    failure = evidence.read_json(fixture['output'], fixture['case'].name+'_worker_failure.json')
    assert failure['evidence_preserved'] and not failure['automatic_retry']
    assert failure['known_collection_artifact_sha256']
    assert (fixture['output']/(fixture['case'].name+'/physics_trace.npz')).exists()
    assert not (fixture['output']/(fixture['case'].name+separate.EXECUTION)).exists()
    with pytest.raises(ValueError, match='no retry'): run(fixture)
    with pytest.raises(ValueError, match='audit failure preserved'): read(fixture, returned)


def test_parent_admission_failure_never_invokes_raw_audit(fixture, worker, monkeypatch):
    def bad(c): raise ValueError('not a fresh audit child')
    monkeypatch.setattr(separate, 'require_audit_child', bad)
    result = run(fixture)
    assert result['status'] == separate.FAILED and worker == []


def test_live_audit_receipt_cannot_be_read_as_completed(fixture, worker, monkeypatch):
    returned = run(fixture)
    monkeypatch.setattr(handoff, 'owner_live', lambda owner:owner['pid'] == 99999998)
    with pytest.raises(ValueError, match='audit worker remains live'): read(fixture, returned)


@pytest.mark.parametrize('field,value', [('same_process_collection_and_raw_audit', True),
    ('original_raw_audit_returned', False), ('parent_verified_audit_zero_exit', True),
    ('wall_s', float('inf')), ('maximum_rss_bytes', True)])
def test_rebound_execution_cannot_change_process_or_measurement_claims(fixture, worker, field, value):
    returned = run(fixture); path = fixture['output']/(fixture['case'].name+separate.EXECUTION)
    execution = json.loads(path.read_text()); execution[field] = value
    path.write_text(json.dumps(execution)+'\n'); returned['separate_audit_execution_sha256'] = digest(path)
    with pytest.raises(ValueError): read(fixture, returned)


def test_changed_closed_audit_log_rejected(fixture, worker):
    returned = run(fixture)
    (fixture['output']/(fixture['case'].name+'_worker.log')).write_text('tampered closed log')
    with pytest.raises(ValueError): read(fixture, returned)


def test_changed_audit_ownership_during_admission_rejected(fixture, worker, monkeypatch):
    returned = run(fixture); states = iter([False, True])
    monkeypatch.setattr(handoff, 'owner_live', lambda owner:
        next(states) if owner['pid'] == 99999998 else False)
    with pytest.raises(ValueError, match='ownership changed during evidence admission'):
        read(fixture, returned)
