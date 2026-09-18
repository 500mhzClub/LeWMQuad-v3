"""Five-stage synthetic evidence; no real runtime artifacts or owner changes."""
from copy import deepcopy

import pytest

from scripts import independent_round_trip_budget_queue_completion_development as queue
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from lewm.tests.test_independent_round_trip_queue_completion_development import synthetic
from lewm.tests.test_independent_round_trip_extended_queue_completion_development import extended


@pytest.fixture
def five(extended, tmp_path, monkeypatch):
    original = queue.original.admit(extended['original'], extended['sha'], sources={}, full=True)
    root = tmp_path/'go2_synthetic_budget_wait_attempt_001'; root.mkdir()
    child = tmp_path/'go2_synthetic_budget_native_attempt_001'; child.mkdir()
    monkeypatch.setattr(queue.budget, 'OUTPUT', root)
    monkeypatch.setattr(queue.budget.native, 'OUTPUT', child)
    monkeypatch.setattr(queue, 'owner_live', lambda owner:False)
    native_input = dict(synthetic_full_input=True)
    write_json(child/'launch.json', dict(source_sha256={}, input_admission=native_input))
    (child/'raw.json').write_text('synthetic raw input\n')
    write_json(child/'result.json', dict(source_sha256={},
        artifact_sha256={n:digest(child/n) for n in ('launch.json', 'raw.json')}))
    completion = dict(native_result_sha256=digest(child/'result.json'), measured_round_trip_successes=0,
        budget_only_preboundary_execution_supported=False, scientific_success_required=False,
        complete_native_worker_and_artifact_roster_verified=True, actual_prefix_finding_reconstructed=True)
    launch = dict(source_sha256={}, boot_id=queue.BOOT, waiter_pid=queue.OWNER['pid'],
        original_owners={s[0]:s[1] for s in queue.budget.PREREQUISITES},
        planned_case=list(queue.budget.native.CASE), automatic_retry=False, source_changes_permitted=False,
        native_workers_while_waiting=0, native_workers_after_original_completion=1)
    inputs = queue.expected_inputs(original)
    write_json(root/'launch.json', launch)
    monkeypatch.setattr(queue, 'LAUNCH_SHA', digest(root/'launch.json'))
    write_json(root/'input_completion.json', inputs); write_json(root/'native_completion.json', completion)
    (root/'events.jsonl').write_text('synthetic closed events\n')
    (root/'native_stdout.log').write_text('synthetic closed child log\n')
    result = dict(status='NO_RGB_DIRECT_EXTENDED_BUDGET_MAZE02_NATIVE_WAIT_V1_COMPLETE',
        source_sha256={}, automatic_retry=False, report=deepcopy(completion),
        artifact_sha256={n:digest(root/n) for n in queue.original.original.WAIT_FILES})
    write_json(root/'result.json', result); calls = []
    def authenticate(sources, ids):
        assert ids == inputs; calls.append('completion'); return deepcopy(completion)
    def admit(batch, identities, tracking, sources):
        assert batch == inputs['batch'] and tracking == inputs['tracking']
        assert identities == {k:inputs[k] for k in ('frontier', 'hold', 'contact')}
        calls.append('full inputs'); return deepcopy(native_input)
    monkeypatch.setattr(queue.budget, 'authenticate_completed', authenticate)
    monkeypatch.setattr(queue.budget.native, 'verify_inputs', lambda launch:calls.append('bound inputs'))
    monkeypatch.setattr(queue.budget.native.inputs, 'admit', admit)
    return dict(original=original, root=root, child=child, completion=completion, launch=launch,
        inputs=inputs, result=result, sha=digest(root/'result.json'), calls=calls)


def admit(env, *, full=True): return queue.admit(env['original'], env['sha'], sources={}, full=full)


def test_complete_negative_five_stage_evidence_does_not_select_policy_or_budget(five):
    result = admit(five)
    assert result['complete_five_stage_queue_authenticated']
    assert result['budget_native_input_admission_fully_reexecuted']
    assert result['budget_completion']['measured_round_trip_successes'] == 0
    assert not result['budget_completion']['budget_only_preboundary_execution_supported']
    assert five['calls'] == ['completion', 'bound inputs', 'full inputs']
    assert not any(result[k] for k in ('native_case_raw_audits_reexecuted', 'final_policy_review_completed',
        'independent_study_budget_selected', 'population_execution_permitted', 'new_layout_sensor_data_consumed'))
    queue.verify_bound(result, {})
    assert five['calls'][-2:] == ['completion', 'bound inputs']


@pytest.mark.parametrize('pid', [queue.OWNER['pid'], *(s[1]['pid'] for s in queue.budget.PREREQUISITES)])
def test_any_live_owner_blocks_before_expensive_predecessor_verification(five, monkeypatch, pid):
    monkeypatch.setattr(queue, 'owner_live', lambda owner:owner['pid'] == pid)
    monkeypatch.setattr(queue.original, 'verify_bound', lambda *a:pytest.fail('live owner must reject first'))
    with pytest.raises(ValueError, match='still live'): admit(five)


@pytest.mark.parametrize('fault', ['status', 'retry', 'missing_log', 'case', 'boot', 'owner',
    'wait_workers', 'child_workers', 'bool_workers', 'source_changes', 'batch_link', 'tracking_link', 'extra_link', 'completion'])
def test_changed_waiter_contract_is_rejected(five, fault):
    result, launch, receipt, completion = [deepcopy(five[k]) for k in ('result', 'launch', 'inputs', 'completion')]
    if fault == 'status': result['status'] = 'INCOMPLETE'
    elif fault == 'retry': result['automatic_retry'] = True
    elif fault == 'missing_log': result['artifact_sha256'].pop('native_stdout.log')
    elif fault == 'case': launch['planned_case'][0] = 'substitute'
    elif fault == 'boot': launch['boot_id'] = 'other'
    elif fault == 'owner': launch['original_owners']['tracking']['pid'] += 1
    elif fault == 'wait_workers': launch['native_workers_while_waiting'] = 1
    elif fault == 'child_workers': launch['native_workers_after_original_completion'] = 2
    elif fault == 'bool_workers': launch['native_workers_after_original_completion'] = True
    elif fault == 'source_changes': launch['source_changes_permitted'] = True
    elif fault == 'batch_link': receipt['batch'] = 'f'*64
    elif fault == 'tracking_link': receipt['tracking'] = 'f'*64
    elif fault == 'extra_link': receipt['replacement'] = 'f'*64
    else: completion['native_result_sha256'] = 'f'*64
    with pytest.raises(ValueError): queue.require_waiter(result, launch, receipt, completion, five['inputs'])


@pytest.mark.parametrize('fault', ['wait_failure', 'native_failure', 'missing_result', 'raw_tamper', 'log_tamper'])
def test_actual_failure_missing_or_modified_files_are_rejected(five, fault):
    if fault == 'wait_failure': (five['root']/'failure.json').write_text('failed')
    elif fault == 'native_failure': (five['child']/'failure.json').write_text('failed')
    elif fault == 'missing_result': (five['root']/'result.json').unlink()
    elif fault == 'raw_tamper': (five['child']/'raw.json').write_text('changed')
    else: (five['root']/'native_stdout.log').write_text('changed')
    with pytest.raises((ValueError, FileNotFoundError)): admit(five)


def test_full_budget_input_admission_required_for_later_bound_checks(five):
    result = admit(five, full=False)
    with pytest.raises(ValueError, match='full budget'): queue.verify_bound(result, {})


def test_full_predecessor_admission_required(five):
    five['original']['tracking_native_input_admission_fully_reexecuted'] = False
    with pytest.raises(ValueError, match='full tracking'): admit(five)


@pytest.mark.parametrize('field', ['final_policy_review_completed', 'independent_study_budget_selected', 'population_execution_permitted'])
def test_completion_cannot_be_relabelled_as_study_authority(five, field):
    result = admit(five); result[field] = True
    with pytest.raises(ValueError, match='same complete five-stage'): queue.verify_bound(result, {})


def test_changed_actual_prefix_completion_is_not_accepted(five, monkeypatch):
    monkeypatch.setattr(queue.budget, 'authenticate_completed', lambda *a:{'different':True})
    with pytest.raises(ValueError, match='must reconstruct'): admit(five)


def test_changed_full_native_input_admission_is_not_accepted(five, monkeypatch):
    monkeypatch.setattr(queue.budget.native.inputs, 'admit', lambda *a:{'different':True})
    with pytest.raises(ValueError, match='input admission changed'): admit(five)


@pytest.mark.parametrize('value', [None, 1, 'true'])
def test_full_scope_requires_an_explicit_boolean(five, value):
    with pytest.raises(ValueError, match='explicit full'): admit(five, full=value)
