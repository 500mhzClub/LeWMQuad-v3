"""Reject a different source model, replay, queue identity or experiment scope."""
from copy import deepcopy
import json
from types import SimpleNamespace
import pytest
from scripts import chained_anchor_native_inputs_development as inputs


def fixture(monkeypatch):
    calls = []
    raw = dict(controller_completion_sha256=inputs.COMPLETION_SHA,
               correction_admission={'original':'model'})
    five = dict(budget_inputs=dict(tracking=inputs.TRACKING_WAIT_SHA),
                all_scientific_failures_retained=True,complete_five_stage_queue_authenticated=True)
    admission = dict(completed_native_queue=dict(original_sustained_input_admission=dict(five_stage_queue_completion=five)))
    receipt = dict(native_result_sha256='native',measured_round_trip_successes=0,scientific_success_required=False,
        complete_native_worker_and_artifact_roster_verified=True,actual_physical_prefix_reconstructed=True,
        final_independent_population_policy_review_performed=False)
    ids = dict(raw=inputs.flow.RAW_RESULT_SHA,sustained='sustained')
    wait = dict(status='DIRECT_FLOW_COMMITMENT_CONTACT_MAZE02_NATIVE_WAIT_V1_COMPLETE',automatic_retry=False,
        artifact_sha256={n:'sha' for n in ('launch.json','events.jsonl','input_completion.json','native_stdout.log','native_completion.json')},
        report=deepcopy(receipt))
    launch = dict(waiter_pid=inputs.FLOW_OWNER['pid'],boot_id=inputs.BOOT,
        original_owners={s[0]:s[1] for s in inputs.flow.PREREQUISITES})
    monkeypatch.setattr(inputs,'owners_ended',lambda: calls.append('owners'))
    monkeypatch.setattr(inputs,'verify',lambda *a: None)
    monkeypatch.setattr(inputs,'verify_artifacts',lambda *a: None)
    monkeypatch.setattr(inputs,'raw_inputs',lambda *a: deepcopy(raw))
    def completed(root,sha,launch_sha,sources):
        if root == inputs.flow.OUTPUT: return deepcopy(wait),deepcopy(launch),{'result.json':sha}
        assert root == inputs.flow.native.OUTPUT
        return {},dict(input_admission=deepcopy(admission)),{'result.json':sha}
    monkeypatch.setattr(inputs,'completed',completed)
    def read(root,name):
        if root == inputs.flow.OUTPUT: return deepcopy(ids if name == 'input_completion.json' else receipt)
        return dict(artifact_sha256={'launch.json':'native-launch'})
    monkeypatch.setattr(inputs,'read_json',read)
    def authenticate(sources,actual):
        assert actual == ids
        calls.append('native_completion')
        return deepcopy(receipt)
    monkeypatch.setattr(inputs.flow,'authenticate_completed',authenticate)
    return raw,five,wait,launch,ids,receipt,calls


def test_original_completed_negative_queue_does_not_adopt_its_policies(monkeypatch):
    *_,calls = fixture(monkeypatch)
    result = inputs.admit(inputs.COMPLETION_SHA,'wait',{})
    assert calls[0] == 'owners'
    assert result['completed_native_queue']['completion']['measured_round_trip_successes'] == 0
    assert result['other_queued_policy_changes_adopted'] is False
    assert result['independent_study_policy_selected'] is False
    inputs.verify_bound(result,{})
    assert calls.count('native_completion') == 1


@pytest.mark.parametrize('fault',['status','retry','roster','owner','boot','raw','receipt','tracking','failures','full_queue','prefix_receipt'])
def test_incomplete_or_different_queue_rejected(monkeypatch,fault):
    raw,five,wait,launch,ids,receipt,calls = fixture(monkeypatch)
    if fault == 'status': wait['status'] = 'INCOMPLETE'
    elif fault == 'retry': wait['automatic_retry'] = True
    elif fault == 'roster': wait['artifact_sha256'].pop('events.jsonl')
    elif fault == 'owner': launch['waiter_pid'] += 1
    elif fault == 'boot': launch['boot_id'] = 'different'
    elif fault == 'raw': ids['raw'] = 'different'
    elif fault == 'receipt': wait['report']['native_result_sha256'] = 'different'
    elif fault == 'tracking': five['budget_inputs']['tracking'] = 'different'
    elif fault == 'failures': five['all_scientific_failures_retained'] = False
    elif fault == 'full_queue': five['complete_five_stage_queue_authenticated'] = False
    else:
        receipt['actual_physical_prefix_reconstructed'] = False
        wait['report'] = deepcopy(receipt)
    with pytest.raises(ValueError): inputs.admit(inputs.COMPLETION_SHA,'wait',{})


@pytest.mark.parametrize('fault',['model','scope','failure','policy','full_completion'])
def test_bound_admission_cannot_change_model_or_scope(monkeypatch,fault):
    fixture(monkeypatch)
    result = inputs.admit(inputs.COMPLETION_SHA,'wait',{})
    if fault == 'model': result['correction_admission']['original'] = 'different'
    elif fault == 'scope': result['other_queued_policy_changes_adopted'] = True
    elif fault == 'failure': result['all_scientific_failures_retained'] = False
    elif fault == 'policy': result['independent_study_policy_selected'] = True
    else: result['completed_native_queue']['queued_completion_reexecuted'] = 1
    with pytest.raises(ValueError): inputs.verify_bound(result,{})


@pytest.mark.parametrize('which',[0,1,2,3])
def test_each_live_owner_blocks_before_raw_input_access(monkeypatch,which):
    owners = (inputs.prefix.completed.OWNER,inputs.TRACKING_OWNER,inputs.FLOW_OWNER)
    monkeypatch.setattr(inputs,'Path',lambda p: SimpleNamespace(read_text=lambda: inputs.BOOT))
    monkeypatch.setattr(inputs,'owner_live',lambda owner: which < 3 and owner == owners[which])
    def earlier():
        if which == 3: raise ValueError('original queue still live')
        pytest.fail('live owner must reject before earlier queue')
    monkeypatch.setattr(inputs.flow.native.inputs,'owners_ended',earlier)
    monkeypatch.setattr(inputs,'raw_inputs',lambda *a: pytest.fail('live owner must reject before data'))
    with pytest.raises(ValueError,match='still live'): inputs.admit(inputs.COMPLETION_SHA,'wait',{})


def raw_fixture(monkeypatch,tmp_path):
    from lewm.tests.test_chained_anchor_native_prefix_development import population
    report = population()[-1]
    actual = {'worker':'sha'}
    ids = {'result.json': inputs.prefix.completed.RESULT_SHA}
    proof = dict(status='CHAINED_ANCHOR_CONTROLLER_COMPLETION_VERIFIED',owner=inputs.prefix.completed.OWNER,
        owner_ended=True,public_packets_reconstructed=854,complete_saved_comparisons_reconstructed=854,
        result_sha256=inputs.prefix.completed.RESULT_SHA,source_sha256={'source':'sha'},
        controller_artifact_sha256=ids,worker_artifact_sha256=deepcopy(actual),report=report)
    p = tmp_path/'proof.json'; p.write_text(json.dumps(proof))
    monkeypatch.setattr(inputs.prefix.completed,'OUTPUT',p)
    monkeypatch.setattr(inputs,'owners_ended',lambda: None)
    monkeypatch.setattr(inputs,'verify',lambda *a: None)
    monkeypatch.setattr(inputs,'verify_artifacts',lambda *a: None)
    native_result = dict(status='NO_RGB_JEPA_DIRECT_FLOW_MAZE02_PILOT_V1_COMPLETE')
    native_launch = dict(planned_case=list(inputs.replay.native.CASE),model_state_sha256=inputs.replay.MODEL_SHA,
        input_admission=dict(full_original_model_and_episode_input_admission_reexecuted=True,
                             correction_admission={'original':'model'}))
    waited = dict(status='NO_RGB_JEPA_DIRECT_FLOW_MAZE02_NATIVE_WAIT_V1_COMPLETE',
        report=dict(native_result_sha256=inputs.TRACKING_SHA,complete_native_worker_and_artifact_roster_verified=True))
    native_ids = deepcopy(actual)
    def completed(root,sha,launch_sha,sources):
        if root == inputs.replay.OUTPUT: return dict(report=deepcopy(report)),dict(observer_artifact_sha256={}),ids
        if root == inputs.replay.native.OUTPUT: return native_result,native_launch,native_ids
        assert root == inputs.tracking.OUTPUT
        return waited,{},{}
    monkeypatch.setattr(inputs,'completed',completed)
    monkeypatch.setattr(inputs.replay.observer,'admit_worker',lambda *a: deepcopy(actual))
    monkeypatch.setattr(inputs.prefix,'read_rows',lambda p: (x for x in ()))
    monkeypatch.setattr(inputs.prefix,'reconstruct',lambda *a: 850)
    monkeypatch.setattr(inputs,'read_json',lambda *a: [])
    return p,proof,native_result,native_launch,waited,native_ids


@pytest.mark.parametrize('fault',[None,'completion','packets','owner','source','model','training','worker','tracking_wait'])
def test_raw_admission_binds_exact_original_controller_and_model(monkeypatch,tmp_path,fault):
    p,proof,result,launch,waited,ids = raw_fixture(monkeypatch,tmp_path)
    sha = inputs.COMPLETION_SHA
    if fault == 'completion': sha = 'different'
    elif fault == 'packets': proof['public_packets_reconstructed'] = 853
    elif fault == 'owner': proof['owner_ended'] = False
    elif fault == 'source': proof['source_sha256']['source'] = 'different'
    elif fault == 'model': launch['model_state_sha256'] = 'different'
    elif fault == 'training': launch['input_admission']['full_original_model_and_episode_input_admission_reexecuted'] = False
    elif fault == 'worker': ids['worker'] = 'different'
    elif fault == 'tracking_wait': waited['report']['native_result_sha256'] = 'different'
    p.write_text(json.dumps(proof))
    if fault:
        with pytest.raises(ValueError): inputs.raw_inputs(sha,{'source':'sha'})
    else:
        admission = inputs.raw_inputs(sha,{'source':'sha'})
        assert admission['correction_admission'] == {'original':'model'}
        assert admission['completed_original_training_admission_reused'] is True
        assert admission['training_data_replayed'] is False
