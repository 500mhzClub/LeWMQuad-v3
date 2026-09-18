"""Require original completed chains, exact model links and preserved failures."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from scripts import direct_flow_commitment_contact_native_inputs_development as inputs


def fixture(monkeypatch):
    calls=[]
    raw=dict(raw_prefix_result_sha256='raw',original_batch_result_sha256='batch',correction_admission={'original':'model'})
    admission=dict(five_stage_queue_completion=dict(budget_inputs=dict(batch='batch',contact=inputs.CONTACT_WAIT_SHA),
        all_scientific_failures_retained=True,complete_five_stage_queue_authenticated=True))
    completion=dict(native_result_sha256='native',measured_round_trip_successes=0,scientific_success_required=False)
    ids=dict(raw=inputs.SUSTAINED_RAW_SHA,budget='budget')
    wait=dict(status='SUSTAINED_HOLD_REORIENTATION_MAZE02_NATIVE_WAIT_V1_COMPLETE',automatic_retry=False,
        artifact_sha256={n:'sha' for n in ('launch.json','events.jsonl','input_completion.json','native_stdout.log','native_completion.json')},
        report=deepcopy(completion))
    launch=dict(waiter_pid=inputs.SUSTAINED_OWNER['pid'],boot_id=inputs.BOOT,
        original_owners={s[0]:s[1] for s in inputs.sustained.PREREQUISITES})
    monkeypatch.setattr(inputs,'owners_ended',lambda:calls.append('owners'))
    monkeypatch.setattr(inputs,'verify',lambda s:None)
    monkeypatch.setattr(inputs,'verify_artifacts',lambda *a:None)
    monkeypatch.setattr(inputs,'raw_inputs',lambda *a:deepcopy(raw))
    def completed(root,sha,launch_sha,sources):
        if root==inputs.sustained.OUTPUT:return deepcopy(wait),deepcopy(launch),{'result.json':sha}
        assert root==inputs.sustained.native.OUTPUT
        return {},dict(input_admission=deepcopy(admission)),{'result.json':sha}
    monkeypatch.setattr(inputs,'completed',completed)
    def read(root,name):
        if root==inputs.sustained.OUTPUT:
            return deepcopy(ids if name=='input_completion.json' else completion)
        return dict(artifact_sha256={'launch.json':'native-launch'})
    monkeypatch.setattr(inputs,'read_json',read)
    def authenticate(sources,actual_ids):
        assert actual_ids==ids;calls.append('native_completion');return deepcopy(completion)
    monkeypatch.setattr(inputs.sustained,'authenticate_completed',authenticate)
    def admit(raw_sha,budget_sha,sources):
        assert raw_sha==inputs.SUSTAINED_RAW_SHA and budget_sha=='budget'
        calls.append('full_admission');return deepcopy(admission)
    monkeypatch.setattr(inputs.sustained.native.inputs,'admit',admit)
    return raw,admission,wait,launch,ids,completion,calls


def test_full_chain_and_negative_native_result_are_preserved(monkeypatch):
    raw,admission,wait,launch,ids,completion,calls=fixture(monkeypatch)
    result=inputs.admit('raw','wait',{})
    assert calls[0]=='owners' and calls.count('full_admission')==1
    queue=result['completed_native_queue']
    assert queue['original_sustained_input_admission']==admission
    assert queue['completion']['measured_round_trip_successes']==0
    assert queue['sustained_input_admission_fully_reexecuted'] is True
    inputs.verify_bound(result,{})
    assert calls.count('full_admission')==1 and calls.count('native_completion')==2


@pytest.mark.parametrize('fault',['status','retry','roster','owner','boot','raw','receipt','full_admission'])
def test_incomplete_or_changed_sustained_chain_rejected(monkeypatch,fault):
    raw,admission,wait,launch,ids,completion,calls=fixture(monkeypatch)
    if fault=='status':wait['status']='INCOMPLETE'
    if fault=='retry':wait['automatic_retry']=True
    if fault=='roster':wait['artifact_sha256'].pop('events.jsonl')
    if fault=='owner':launch['waiter_pid']+=1
    if fault=='boot':launch['boot_id']='changed'
    if fault=='raw':ids['raw']='different'
    if fault=='receipt':wait['report']['native_result_sha256']='different'
    if fault=='full_admission':monkeypatch.setattr(inputs.sustained.native.inputs,'admit',lambda *a:{'changed':True})
    with pytest.raises(ValueError):inputs.admit('raw','wait',{})


@pytest.mark.parametrize('fault',['batch','contact','model','queue_identity','scope','full_flag','failures'])
def test_bound_inputs_cannot_change_model_queue_or_scope(monkeypatch,fault):
    raw,admission,*_=fixture(monkeypatch);result=inputs.admit('raw','wait',{})
    if fault=='batch':result['original_batch_result_sha256']='different'
    if fault=='contact':result['completed_native_queue']['original_sustained_input_admission']['five_stage_queue_completion']['budget_inputs']['contact']='different'
    if fault=='model':result['correction_admission']['original']='different'
    if fault=='queue_identity':result['sustained_wait_result_sha256']='different'
    if fault=='scope':result['other_queued_policy_changes_adopted']=True
    if fault=='full_flag':result['completed_native_queue']['sustained_input_admission_fully_reexecuted']=1
    if fault=='failures':result['all_scientific_failures_retained']=False
    with pytest.raises(ValueError):inputs.verify_bound(result,{})


@pytest.mark.parametrize('live',['raw','sustained','queue'])
def test_every_live_owner_blocks_before_raw_input_access(monkeypatch,live):
    monkeypatch.setattr(inputs,'Path',lambda p:SimpleNamespace(read_text=lambda:inputs.BOOT))
    monkeypatch.setattr(inputs,'owner_live',lambda owner:owner==inputs.prefix.OWNER if live=='raw' else owner==inputs.SUSTAINED_OWNER if live=='sustained' else False)
    def queue():
        if live=='queue':raise ValueError('original queue still live')
        pytest.fail('earlier live owner should reject first')
    monkeypatch.setattr(inputs.sustained.native.inputs,'owners_ended',queue)
    monkeypatch.setattr(inputs,'raw_inputs',lambda *a:pytest.fail('live queue must block inputs'))
    with pytest.raises(ValueError,match='still live'):inputs.admit('raw','wait',{})


@pytest.mark.parametrize('fault',[None,'worker_inputs','model','worker','wait_result','original_prefix'])
def test_raw_prefix_is_bound_to_same_completed_contact_worker(monkeypatch,fault):
    monkeypatch.setattr(inputs,'owners_ended',lambda:None);monkeypatch.setattr(inputs,'verify',lambda *a:None)
    actual={'raw_worker':'sha'}
    raw_launch=dict(observer_artifact_sha256={'observer':'sha'},input_artifact_sha256=deepcopy(actual))
    native_launch=dict(model_state_sha256=inputs.replay.MODEL_SHA,input_admission=dict(
        prefix_result_sha256='prefix',adapter_batch_result_sha256='batch',correction_admission={'original':'model'}))
    native_result=dict(prospective_prefix_result_sha256='prefix')
    waited=dict(report=dict(native_result_sha256=inputs.CONTACT_SHA))
    native_ids={inputs.contact.native.CASE[0]+'_worker_terminal.json':inputs.replay.observer.probe.diagnosis.WORKER_SHA}
    report=dict(model_state_sha256=inputs.replay.MODEL_SHA)
    if fault=='worker_inputs':raw_launch['input_artifact_sha256']={}
    if fault=='model':report['model_state_sha256']='different'
    if fault=='worker':native_ids[next(iter(native_ids))]='different'
    if fault=='wait_result':waited['report']['native_result_sha256']='different'
    if fault=='original_prefix':native_result['prospective_prefix_result_sha256']='different'
    def completed(root,sha,launch_sha,sources):
        if root==inputs.replay.OUTPUT:return {},raw_launch,{'result.json':sha}
        if root==inputs.contact.native.OUTPUT:return native_result,native_launch,native_ids
        assert root==inputs.contact.OUTPUT
        return waited,{},{}
    monkeypatch.setattr(inputs,'completed',completed)
    monkeypatch.setattr(inputs.prefix,'admit_prefix',lambda *a:deepcopy(report))
    monkeypatch.setattr(inputs.replay,'verify_inputs',lambda *a:deepcopy(actual))
    if fault:
        with pytest.raises(ValueError):inputs.raw_inputs('raw',{})
    else:
        result=inputs.raw_inputs('raw',{})
        assert result['original_batch_result_sha256']=='batch'
        assert result['correction_admission']=={'original':'model'}
        assert result['prefix_report']==report
