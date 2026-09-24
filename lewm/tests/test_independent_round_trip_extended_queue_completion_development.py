"""Synthetic four-stage completion chain; no original datasets or processes."""
from copy import deepcopy
import pytest

from scripts import independent_round_trip_extended_queue_completion_development as queue
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from lewm.tests.test_independent_round_trip_queue_completion_development import synthetic


@pytest.fixture
def extended(synthetic, tmp_path, monkeypatch):
    original = queue.original.admit(synthetic['identities'],
        adapter_batch_result_sha256=synthetic['batch_sha'], sources={}, full=True)
    root = tmp_path/'go2_synthetic_tracking_wait_attempt_001'; root.mkdir()
    child = tmp_path/'go2_synthetic_tracking_native_attempt_001'; child.mkdir()
    monkeypatch.setattr(queue.tracking, 'OUTPUT', root)
    monkeypatch.setattr(queue.tracking.native, 'OUTPUT', child)
    monkeypatch.setattr(queue, 'owner_live', lambda owner: False)
    write_json(child/'launch.json', dict(source_sha256={}, fixture=True))
    (child/'raw.json').write_text('synthetic raw evidence\n')
    native_result = dict(source_sha256={}, artifact_sha256={
        n: digest(child/n) for n in ('launch.json', 'raw.json')})
    write_json(child/'result.json', native_result)
    completion = dict(native_result_sha256=digest(child/'result.json'),
        measured_round_trip_successes=0, scientific_success_required=False,
        complete_native_worker_and_artifact_roster_verified=True)
    launch = dict(source_sha256={}, boot_id=queue.BOOT, waiter_pid=queue.OWNER['pid'],
        original_owners={s[0]: s[1] for s in queue.tracking.PREREQUISITES},
        planned_case=list(queue.tracking.native.CASE),
        reviewed_prefix_artifact_sha256=queue.tracking.PREFIX_BINDINGS,
        automatic_retry=False, source_changes_permitted=False,
        native_workers_while_waiting=0, native_workers_after_original_completion=1)
    inputs = queue.expected_inputs(original)
    write_json(root/'launch.json', launch)
    monkeypatch.setattr(queue, 'LAUNCH_SHA', digest(root/'launch.json'))
    write_json(root/'input_completion.json', inputs)
    write_json(root/'native_completion.json', completion)
    (root/'events.jsonl').write_text('synthetic closed event log\n')
    (root/'native_stdout.log').write_text('synthetic closed child log\n')
    result = dict(status='NO_RGB_JEPA_DIRECT_FLOW_MAZE02_NATIVE_WAIT_V1_COMPLETE',
        source_sha256={}, automatic_retry=False, report=deepcopy(completion),
        artifact_sha256={n: digest(root/n) for n in queue.original.WAIT_FILES})
    write_json(root/'result.json', result)
    calls = []
    def authenticate(sources, ids):
        assert ids == inputs
        calls.append(('completion', deepcopy(ids)))
        return deepcopy(completion)
    def verify_inputs(launch, *, full=False):
        calls.append(('inputs', full))
    monkeypatch.setattr(queue.tracking, 'authenticate_completed', authenticate)
    monkeypatch.setattr(queue.tracking.native, 'verify_inputs', verify_inputs)
    return dict(original=original, root=root, child=child, completion=completion,
        launch=launch, inputs=inputs, result=result, sha=digest(root/'result.json'), calls=calls)


def admit(env, *, full=True):
    return queue.admit(env['original'], env['sha'], sources={}, full=full)


def test_complete_four_stage_negative_population_retained(extended):
    result = admit(extended)
    assert result['complete_extended_queue_authenticated']
    assert result['tracking_native_input_admission_fully_reexecuted']
    assert result['original_native_input_admissions_previously_fully_reexecuted']
    assert result['tracking_completion']['measured_round_trip_successes'] == 0
    assert result['tracking_inputs']['prefix'] == queue.PREFIX_SHA
    assert extended['calls'] == [('completion', extended['inputs']), ('inputs', True)]
    assert not any(result[k] for k in ('native_case_raw_audits_reexecuted',
        'final_policy_review_completed', 'population_execution_permitted', 'new_layout_sensor_data_consumed'))
    queue.verify_bound(result, {})
    assert extended['calls'][-1] == ('inputs', False)


@pytest.mark.parametrize('pid', [queue.OWNER['pid'], *(s[1]['pid'] for s in queue.tracking.PREREQUISITES)])
def test_any_live_owner_blocks_before_predecessor_artifact_reads(extended, monkeypatch, pid):
    monkeypatch.setattr(queue, 'owner_live', lambda owner: owner['pid'] == pid)
    monkeypatch.setattr(queue.original, 'verify_bound', lambda *a: pytest.fail('live owner must reject first'))
    with pytest.raises(ValueError, match='still live'): admit(extended)


def test_reused_pid_is_not_completion(extended, monkeypatch):
    def changed(owner): raise ValueError('original owner identity changed')
    monkeypatch.setattr(queue, 'owner_live', changed)
    with pytest.raises(ValueError, match='identity changed'): admit(extended)


@pytest.mark.parametrize('fault', ['status', 'retry', 'source_changes', 'case', 'boot', 'owner',
    'prefix_artifact', 'missing_log', 'wait_workers', 'child_workers', 'boolean_workers',
    'prefix_link', 'batch_link', 'queue_link', 'extra_link', 'completion'])
def test_waiter_contract_rejects_changed_receipts(extended, fault):
    result, launch, receipt, completion = [deepcopy(extended[k]) for k in ('result','launch','inputs','completion')]
    if fault == 'status': result['status'] = 'INCOMPLETE'
    elif fault == 'retry': result['automatic_retry'] = True
    elif fault == 'source_changes': launch['source_changes_permitted'] = True
    elif fault == 'case': launch['planned_case'][0] = 'replacement'
    elif fault == 'boot': launch['boot_id'] = 'replacement'
    elif fault == 'owner': launch['original_owners']['prefix']['pid'] += 1
    elif fault == 'prefix_artifact': launch['reviewed_prefix_artifact_sha256']['report.json'] = 'f'*64
    elif fault == 'missing_log': result['artifact_sha256'].pop('native_stdout.log')
    elif fault == 'wait_workers': launch['native_workers_while_waiting'] = 1
    elif fault == 'child_workers': launch['native_workers_after_original_completion'] = 2
    elif fault == 'boolean_workers': launch['native_workers_after_original_completion'] = True
    elif fault == 'prefix_link': receipt['prefix'] = 'f'*64
    elif fault == 'batch_link': receipt['batch'] = 'f'*64
    elif fault == 'queue_link': receipt['contact'] = 'f'*64
    elif fault == 'extra_link': receipt['replacement'] = 'f'*64
    else: completion['native_result_sha256'] = 'f'*64
    with pytest.raises(ValueError):
        queue.require_waiter(result, launch, receipt, completion, extended['inputs'])


@pytest.mark.parametrize('target', ['wait_failure','native_failure','missing_result','wait_log','raw_data'])
def test_actual_artifact_failure_or_tamper_rejected(extended, target):
    if target == 'wait_failure': (extended['root']/'failure.json').write_text('failed')
    elif target == 'native_failure': (extended['child']/'failure.json').write_text('failed')
    elif target == 'missing_result': (extended['root']/'result.json').unlink()
    elif target == 'wait_log': (extended['root']/'native_stdout.log').write_text('changed')
    else: (extended['child']/'raw.json').write_text('changed')
    with pytest.raises((ValueError, FileNotFoundError)): admit(extended)


def test_original_full_admission_required(extended):
    extended['original']['original_native_input_admissions_fully_reexecuted'] = False
    with pytest.raises(ValueError, match='original full queue input admission'): admit(extended)


def test_original_queue_failure_propagates(extended, monkeypatch):
    def reject(*a): raise ValueError('original queue evidence changed')
    monkeypatch.setattr(queue.original, 'verify_bound', reject)
    with pytest.raises(ValueError, match='queue evidence changed'): admit(extended)


def test_tracking_completion_must_reconstruct(extended, monkeypatch):
    monkeypatch.setattr(queue.tracking, 'authenticate_completed', lambda *a: {'different': True})
    with pytest.raises(ValueError, match='reconstruct exactly'): admit(extended)


def test_tracking_full_input_failure_propagates(extended, monkeypatch):
    def reject(*a, **k): raise ValueError('tracking input admission failed')
    monkeypatch.setattr(queue.tracking.native, 'verify_inputs', reject)
    with pytest.raises(ValueError, match='tracking input admission failed'): admit(extended)


@pytest.mark.parametrize('value', [None, 1, 'yes'])
def test_explicit_boolean_full_scope(extended, value):
    with pytest.raises(ValueError, match='explicit full'): admit(extended, full=value)


@pytest.mark.parametrize('value', [None, 'bad', 'F'*64])
def test_exact_result_sha_required(extended, value):
    extended['sha'] = value
    with pytest.raises(ValueError, match='SHA-256'): admit(extended)


def test_bound_check_requires_full_tracking_admission(extended):
    result = admit(extended, full=False)
    with pytest.raises(ValueError, match='original full tracking'): queue.verify_bound(result, {})


@pytest.mark.parametrize('field', ['population_execution_permitted','final_policy_review_completed'])
def test_completion_cannot_be_relabelled_as_execution_or_policy_approval(extended, field):
    result = admit(extended); result[field] = True
    with pytest.raises(ValueError, match='same complete extended'): queue.verify_bound(result, {})
