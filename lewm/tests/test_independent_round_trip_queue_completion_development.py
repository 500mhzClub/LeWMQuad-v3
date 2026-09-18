"""Synthetic queue files and substituted original verifiers; no native data."""
from copy import deepcopy
import json
import numpy as np
import pytest

from scripts import independent_round_trip_queue_completion_development as queue
from scripts import navigation_artifact_root_development as guard
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from lewm.tests.test_independent_round_trip_population_readout_development import fixture as population_fixture


def replace(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, 'BASE', tmp_path)
    monkeypatch.setattr(queue, 'owner_live', lambda owner: False)
    jobs = []; data = []; calls = []; identities = {key:'0'*64 for key in ('frontier','hold','contact')}; batch_sha = 'a'*64
    base_records, base_reports, contacts = population_fixture()
    for index, (key, waiter, _, status, pid, created) in enumerate(queue.JOBS):
        root = tmp_path/f'go2_synthetic_queue_{key}_wait_attempt_001'; root.mkdir()
        child = tmp_path/f'go2_synthetic_queue_{key}_native_attempt_001'; child.mkdir()
        monkeypatch.setattr(waiter, 'OUTPUT', root); monkeypatch.setattr(waiter.native, 'OUTPUT', child)
        native = waiter.native; name = native.CASE[0]
        directory = child/name; directory.mkdir()
        collection = deepcopy(base_records[0]['collection']); report = deepcopy(base_reports[0])
        write_json(directory/'result.json', collection)
        (directory/'raw.json').write_text('synthetic raw placeholder\n')
        np.savez(directory/'physics_trace.npz', physics_contact=contacts[0])
        readout = native.case_readout(report, collection, contacts[0])
        record = dict(case=name, collection=collection, readout=readout, verified_round_trip=False)
        (child/(name+'_worker.log')).write_text('synthetic closed worker log\n')
        record['worker_log_sha256'] = digest(child/(name+'_worker.log'))
        for suffix, value in [('_audit.json', report), ('_prefix_comparison.json', {'synthetic':True}),
                ('_readout.json', readout), ('_worker_terminal.json', record)]:
            write_json(child/(name+suffix), value)
        write_json(child/'launch.json', {'source_sha256':{}, 'synthetic_fixture':True})
        names = ['launch.json', *(name+'/'+n for n in ('result.json','raw.json','physics_trace.npz')),
            *(name+s for s in ('_audit.json','_prefix_comparison.json','_readout.json','_worker_terminal.json','_worker.log'))]
        result = dict(source_sha256={}, artifact_sha256={n:digest(child/n) for n in names},
            conditions=[record], measured_round_trip_successes=0)
        write_json(child/'result.json', result)
        native_sha = digest(child/'result.json')
        completion = dict(native_result_sha256=native_sha, measured_round_trip_successes=0,
            scientific_success_required=False, synthetic_original_verifier=True)
        launch = dict(source_sha256={}, waiter_pid=pid, planned_case=list(native.CASE),
            automatic_retry=False, source_changes_permitted=False,
            native_workers_while_waiting=0, native_workers_after_original_completion=1)
        write_json(root/'launch.json', launch)
        receipt = queue.expected_receipt(index, identities, batch_sha)
        write_json(root/'input_completion.json', receipt); write_json(root/'native_completion.json', completion)
        (root/'events.jsonl').write_text('synthetic events\n'); (root/'native_stdout.log').write_text('synthetic child stdout\n')
        waited = dict(status=status, automatic_retry=False, source_sha256={}, report=completion,
            artifact_sha256={n:digest(root/n) for n in queue.WAIT_FILES})
        write_json(root/'result.json', waited); identities[key] = digest(root/'result.json')
        jobs.append((key, waiter, digest(root/'launch.json'), status, pid, created))
        row = dict(root=root, child=child, waiter=waiter, key=key, collection=collection, report=report,
            record=record, result=result, launch=launch, waited=waited, completion=completion, receipt=receipt)
        data.append(row)
        def authenticate(sources, receipt, *, selected=row):
            calls.append(('completion', selected['key'])); return deepcopy(selected['completion'])
        def inputs(launch, *, full=False, selected=row):
            calls.append(('inputs', selected['key'], full))
        monkeypatch.setattr(waiter, 'authenticate_completed', authenticate)
        monkeypatch.setattr(native, 'verify_inputs', inputs)
        monkeypatch.setattr(native, 'artifacts', lambda layout, collection: ['result.json','raw.json','physics_trace.npz'])
    monkeypatch.setattr(queue, 'JOBS', tuple(jobs))
    return dict(data=data, identities=identities, batch_sha=batch_sha, calls=calls)


def test_complete_negative_queue_is_admitted_without_policy_or_execution_claim(synthetic):
    env = synthetic
    result = queue.admit(env['identities'], adapter_batch_result_sha256=env['batch_sha'], sources={}, full=True)
    assert [row['stage'] for row in result['completed']] == ['frontier','hold','contact']
    assert all(row['measured_round_trip_successes'] == 0 and not row['scientific_success_required']
        for row in result['completed'])
    assert result['complete_original_native_queue_authenticated'] and result['all_scientific_failures_retained']
    assert result['original_native_input_admissions_fully_reexecuted']
    assert not any(result[k] for k in ('native_case_raw_audits_reexecuted', 'final_policy_review_completed',
        'population_execution_permitted','new_layout_sensor_data_consumed'))
    assert env['calls'] == [(operation, key, True) if operation == 'inputs' else (operation, key)
        for key in ('frontier','hold','contact') for operation in ('completion','inputs')]
    queue.verify_bound(result, {})
    assert env['calls'][-1] == ('inputs','contact',False)


@pytest.mark.parametrize('index', range(3))
def test_live_original_owner_blocks_before_any_completion_read(synthetic, monkeypatch, index):
    pid = queue.JOBS[index][4]
    monkeypatch.setattr(queue, 'owner_live', lambda owner: owner['pid'] == pid)
    def forbidden(*args, **kwargs): raise AssertionError('live queue evidence must not be treated as complete')
    monkeypatch.setattr(queue, 'completed', forbidden)
    with pytest.raises(ValueError, match='still live'):
        queue.admit(synthetic['identities'], adapter_batch_result_sha256=synthetic['batch_sha'], sources={})


def test_reused_owner_identity_is_not_treated_as_completion(synthetic, monkeypatch):
    def changed(owner): raise ValueError('original owner process identity changed')
    monkeypatch.setattr(queue, 'owner_live', changed)
    with pytest.raises(ValueError, match='identity changed'):
        queue.admit(synthetic['identities'], adapter_batch_result_sha256=synthetic['batch_sha'], sources={})


@pytest.mark.parametrize('fault', ['missing','extra','bad_hash','bad_batch'])
def test_exact_all_three_result_identities_required(synthetic, fault):
    env = synthetic; ids = dict(env['identities']); batch = env['batch_sha']
    if fault == 'missing': ids.pop('hold')
    elif fault == 'extra': ids['replacement'] = 'a'*64
    elif fault == 'bad_hash': ids['hold'] = 'not-a-sha'
    else: batch = None
    with pytest.raises(ValueError): queue.admit(ids, adapter_batch_result_sha256=batch, sources={})


@pytest.mark.parametrize('index', range(3))
@pytest.mark.parametrize('fault', ['chain','raw_prefix','status','missing_file','retry','child','completion'])
def test_ordered_waiter_contract_rejects_substituted_evidence(synthetic, index, fault):
    env = synthetic; row = env['data'][index]
    result, launch, receipt, completion = [deepcopy(row[k]) for k in ('waited','launch','receipt','completion')]
    if fault == 'chain':
        key = ('adapter_batch_result_sha256','frontier_wait_result_sha256','prior_native_wait_result_sha256')[index]
        receipt[key] = 'f'*64
    elif fault == 'raw_prefix': receipt[queue.RAW_RESULTS[index][0]] = 'f'*64
    elif fault == 'status': result['status'] = 'INCOMPLETE'
    elif fault == 'missing_file': result['artifact_sha256'].pop('events.jsonl')
    elif fault == 'retry': launch['automatic_retry'] = True
    elif fault == 'child': launch['planned_case'][0] = 'substitute'
    else: completion['native_result_sha256'] = 'f'*64
    with pytest.raises(ValueError): queue.require_waiter(index, result, launch, receipt, completion,
        env['identities'], env['batch_sha'])


@pytest.mark.parametrize('target', ['waiter_failure','native_failure','raw_tamper','log_tamper','waiter_tamper'])
def test_original_failures_and_changed_artifacts_reject(synthetic, target):
    env = synthetic; row = env['data'][0]; name = row['waiter'].native.CASE[0]
    if target == 'waiter_failure': (row['root']/'failure.json').write_text('original failed')
    elif target == 'native_failure': (row['child']/'failure.json').write_text('original failed')
    elif target == 'raw_tamper': (row['child']/(name+'/raw.json')).write_text('changed raw')
    elif target == 'log_tamper': (row['child']/(name+'_worker.log')).write_text('changed log')
    else: (row['root']/'events.jsonl').write_text('changed events')
    with pytest.raises(ValueError): queue.admit(env['identities'], adapter_batch_result_sha256=env['batch_sha'], sources={})


@pytest.mark.parametrize('fault', ['missing_raw_binding','inconsistent_log','inconsistent_collection','false_readout'])
def test_rebound_result_cannot_hide_missing_or_inconsistent_native_evidence(synthetic, fault):
    env = synthetic; row = env['data'][0]; name = row['waiter'].native.CASE[0]
    if fault == 'missing_raw_binding': row['result']['artifact_sha256'].pop(name+'/raw.json')
    elif fault == 'inconsistent_log': row['record']['worker_log_sha256'] = 'f'*64
    elif fault == 'inconsistent_collection': row['record']['collection'] = {'wrong':'collection'}
    else: row['record']['readout']['native_contact_samples'] = 12
    replace(row['child']/(name+'_worker_terminal.json'), row['record'])
    row['result']['artifact_sha256'][name+'_worker_terminal.json'] = digest(row['child']/(name+'_worker_terminal.json'))
    replace(row['child']/'result.json', row['result'])
    row['completion']['native_result_sha256'] = digest(row['child']/'result.json')
    replace(row['root']/'native_completion.json', row['completion'])
    row['waited']['artifact_sha256']['native_completion.json'] = digest(row['root']/'native_completion.json')
    replace(row['root']/'result.json', row['waited']); env['identities']['frontier'] = digest(row['root']/'result.json')
    with pytest.raises(ValueError): queue.authenticate_job(0, env['identities'], env['batch_sha'], {}, full=False)


def test_original_completion_verifier_must_reproduce_saved_report(synthetic, monkeypatch):
    env = synthetic
    monkeypatch.setattr(queue.JOBS[0][1], 'authenticate_completed', lambda sources, receipt: {'different':True})
    with pytest.raises(ValueError, match='must reproduce'):
        queue.admit(env['identities'], adapter_batch_result_sha256=env['batch_sha'], sources={})


def test_bound_verification_requires_full_initial_input_verifiers(synthetic):
    env = synthetic; result = queue.admit(env['identities'], adapter_batch_result_sha256=env['batch_sha'], sources={})
    with pytest.raises(ValueError, match='original full queue input admission'): queue.verify_bound(result, {})


def test_bound_verification_does_not_accept_a_policy_approval_flag(synthetic):
    env = synthetic
    result = queue.admit(env['identities'], adapter_batch_result_sha256=env['batch_sha'], sources={}, full=True)
    result['final_policy_review_completed'] = True
    with pytest.raises(ValueError, match='same complete original ordered queue evidence'): queue.verify_bound(result, {})


@pytest.mark.parametrize('field', ['native_workers_while_waiting','native_workers_after_original_completion'])
def test_worker_counts_require_integers_not_boolean_aliases(synthetic, field):
    env = synthetic; row = env['data'][0]; launch = deepcopy(row['launch'])
    launch[field] = bool(launch[field])
    with pytest.raises(ValueError): queue.require_waiter(0, row['waited'], launch, row['receipt'], row['completion'],
        env['identities'], env['batch_sha'])


@pytest.mark.parametrize('full', [1, 'yes', None])
def test_full_input_scope_must_be_explicit_boolean(synthetic, full):
    with pytest.raises(ValueError, match='explicit full-input-verification boolean'):
        queue.admit(synthetic['identities'], adapter_batch_result_sha256=synthetic['batch_sha'], sources={}, full=full)


def test_original_native_input_failure_is_not_ignored(synthetic, monkeypatch):
    def reject(launch, *, full=False): raise ValueError('original native input admission failed')
    monkeypatch.setattr(queue.JOBS[0][1].native, 'verify_inputs', reject)
    with pytest.raises(ValueError, match='original native input admission failed'):
        queue.admit(synthetic['identities'], adapter_batch_result_sha256=synthetic['batch_sha'], sources={}, full=True)
