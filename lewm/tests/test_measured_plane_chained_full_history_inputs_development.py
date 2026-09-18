"""Future chained replay admission rejects live, incomplete and altered input."""
from copy import deepcopy
import json

import pytest

from scripts import measured_plane_chained_full_history_inputs_development as inputs


def fixture(success=False):
    launch = dict(source_sha256={'fixed': 'source'}, input_admission=dict(
        chained_wait_result_sha256='wait', controller_replay_result_sha256='replay'))
    record = dict(verified_round_trip=success)
    result = dict(status='MEASURED_PLANE_CHAINED_MAZE02_V1_COMPLETE',
        source_sha256=deepcopy(launch['source_sha256']), artifact_sha256={'launch.json': inputs.LAUNCH_SHA},
        conditions=[record], chained_wait_result_sha256='wait', controller_replay_result_sha256='replay',
        learned_result_sha256=inputs.native.inputs.LEARNED_RESULT_SHA,
        measured_round_trip_successes=int(success), reused_layout_executions=1,
        new_independent_layout_executions=0, measured_plane_constrained_estimator=True,
        chained_anchor_reacquisition_enabled=True, original_bridge_allowance_unchanged=True,
        single_pass_timing_change_adopted=False, automatic_retry=False, model_training=False,
        navigation_qualified=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False)
    return result, launch


@pytest.mark.parametrize('success', [False, True])
def test_completed_negative_and_positive_outcomes_are_both_admissible(success):
    result, launch = fixture(success)
    assert inputs.require_result(result, launch) is result['conditions'][0]


@pytest.mark.parametrize('key,value', [
    ('status', 'RUNNING'), ('chained_wait_result_sha256', 'changed'),
    ('controller_replay_result_sha256', 'changed'), ('learned_result_sha256', 'changed'),
    ('reused_layout_executions', True), ('new_independent_layout_executions', 1),
    ('original_bridge_allowance_unchanged', False), ('single_pass_timing_change_adopted', True),
    ('automatic_retry', True), ('model_training', True), ('navigation_qualified', True),
    ('measured_round_trip_successes', True), ('measured_round_trip_successes', 1)])
def test_changed_scope_ancestry_or_outcome_counts_are_rejected(key, value):
    result, launch = fixture(); result[key] = value
    with pytest.raises(ValueError): inputs.require_result(result, launch)


@pytest.mark.parametrize('fault', ['sources', 'launch', 'population', 'bool_success'])
def test_complete_result_structure_and_boolean_physical_outcome_are_required(fault):
    result, launch = fixture()
    if fault == 'sources': result['source_sha256']['fixed'] = 'changed'
    elif fault == 'launch': result['artifact_sha256']['launch.json'] = 'changed'
    elif fault == 'population': result['conditions'].append(deepcopy(result['conditions'][0]))
    else: result['conditions'][0]['verified_round_trip'] = 0
    with pytest.raises(ValueError): inputs.require_result(result, launch)


@pytest.mark.parametrize('live', ['parent', 'worker'])
def test_live_owner_prevents_access_to_future_result(monkeypatch, live):
    monkeypatch.setattr(inputs, 'native_launch', lambda: {})
    monkeypatch.setattr(inputs.run, 'owner_live', lambda owner: live == 'parent')
    monkeypatch.setattr(inputs, 'worker_live', lambda: live == 'worker')
    monkeypatch.setattr(inputs.run, 'read_json', lambda *args: pytest.fail('future result read while owner live'))
    with pytest.raises(ValueError, match='must end'):
        inputs.admit('a'*64, {})


def synthetic_admission(tmp_path, monkeypatch):
    name = inputs.native.CASE[0]; result, launch = fixture()
    record = result['conditions'][0]
    record.update(collection=dict(decisions=5, schedule_terminal='MISSION_TICK_BUDGET_EXHAUSTED'),
        artifact_sha256={name+'/context_decisions.jsonl.gz': 'context'},
        worker_log_sha256='worker_log', readout={'outcome': 'negative'},
        prefix_comparison={'physical_prefix': 'reconstructed'}, model_state_sha256='model')
    ids = result['artifact_sha256']
    for suffix in ('_worker_terminal.json', '_worker.log', '_audit.json', '_prefix_comparison.json', '_readout.json'):
        ids[name+suffix] = suffix
    ids.update(record['artifact_sha256']); ids[name+'_worker.log'] = 'worker_log'
    ids[name+'/result.json'] = 'collection'; ids['resource_monitor.jsonl'] = 'resources'
    saved = {(tmp_path, 'result.json'): result,
        (tmp_path, name+'_worker_terminal.json'): deepcopy(record),
        (tmp_path/name, 'result.json'): deepcopy(record['collection']),
        (tmp_path, name+'_audit.json'): {'synthetic_audit': True},
        (tmp_path, name+'_readout.json'): deepcopy(record['readout']),
        (tmp_path, name+'_prefix_comparison.json'): deepcopy(record['prefix_comparison'])}
    checks = []
    monkeypatch.setattr(inputs.native, 'OUTPUT', tmp_path)
    monkeypatch.setattr(inputs, 'native_launch', lambda: deepcopy(launch))
    monkeypatch.setattr(inputs.run, 'owner_live', lambda owner: False)
    monkeypatch.setattr(inputs, 'worker_live', lambda: False)
    monkeypatch.setattr(inputs.run, 'read_json', lambda root, leaf: deepcopy(saved[(root,leaf)]))
    monkeypatch.setattr(inputs.run, 'verify_artifacts', lambda root, roster: checks.append(('artifacts', dict(roster))))
    monkeypatch.setattr(inputs.run, 'verify', lambda sources: checks.append(('sources', dict(sources))))
    monkeypatch.setattr(inputs.native.pipeline, 'artifacts', lambda layout, collection: ['context_decisions.jsonl.gz','result.json'])
    monkeypatch.setattr(inputs.native, 'require_worker', lambda record, audit: checks.append(('worker', deepcopy(record), deepcopy(audit))))
    return saved, checks, name, result, launch


def test_admission_checks_full_roster_receipts_and_prefix_then_reauthenticates(tmp_path, monkeypatch):
    saved, checks, name, result, launch = synthetic_admission(tmp_path, monkeypatch)
    receipt = inputs.admit('a'*64, launch['source_sha256'])
    assert receipt['frames'] == 5 and receipt['native_case'] == name
    assert receipt['original_verified_round_trip'] is False
    assert receipt['original_scientific_success_required'] is False
    assert receipt['original_context_sha256'] == 'context'
    assert receipt['original_physical_prefix_reconstructed'] is True
    assert receipt['original_controller_replayed'] is False
    assert [c[0] for c in checks] == ['artifacts', 'artifacts', 'worker', 'sources', 'artifacts']
    assert checks[-1][1] == result['artifact_sha256'] | {'result.json': 'a'*64}


@pytest.mark.parametrize('fault', ['missing_raw', 'worker_receipt', 'collection_receipt',
    'worker_binding', 'worker_log', 'readout', 'prefix', 'source_union', 'failure', 'result_sha'])
def test_admission_rejects_missing_or_changed_input_evidence(tmp_path, monkeypatch, fault):
    saved, checks, name, result, launch = synthetic_admission(tmp_path, monkeypatch)
    sources = deepcopy(launch['source_sha256']); sha = 'a'*64
    if fault == 'missing_raw': del result['artifact_sha256'][name+'/context_decisions.jsonl.gz']
    elif fault == 'worker_receipt': saved[(tmp_path,name+'_worker_terminal.json')]['verified_round_trip'] = True
    elif fault == 'collection_receipt': saved[(tmp_path/name,'result.json')]['decisions'] = 4
    elif fault == 'worker_binding': result['artifact_sha256'][name+'/context_decisions.jsonl.gz'] = 'changed'
    elif fault == 'worker_log': result['artifact_sha256'][name+'_worker.log'] = 'changed'
    elif fault == 'readout': saved[(tmp_path,name+'_readout.json')]['outcome'] = 'changed'
    elif fault == 'prefix': saved[(tmp_path,name+'_prefix_comparison.json')]['physical_prefix'] = 'changed'
    elif fault == 'source_union': sources['fixed'] = 'changed'
    elif fault == 'failure': (tmp_path/'failure.json').write_text(json.dumps({'failed':True}))
    else: sha = 'placeholder'
    with pytest.raises(ValueError): inputs.admit(sha, sources)
    if fault in ('failure','result_sha'): assert not checks
