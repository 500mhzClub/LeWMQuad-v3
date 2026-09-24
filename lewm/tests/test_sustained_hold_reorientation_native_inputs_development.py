"""Synthetic admission wiring and mixed-identity rejection."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from scripts import sustained_hold_reorientation_native_inputs_development as inputs


def fixture(monkeypatch):
    calls=[]
    raw=dict(raw_prefix_result_sha256='raw',original_batch_result_sha256='batch',correction_admission={'model':'original'})
    five=dict(budget_inputs=dict(batch='batch',hold=inputs.replay.saved.prior.WAIT_SHA),
        all_scientific_failures_retained=True,complete_five_stage_queue_authenticated=True,
        budget_native_input_admission_fully_reexecuted=True,budget_wait_result_sha256='budget')
    monkeypatch.setattr(inputs,'owners_ended',lambda:calls.append('owners'))
    monkeypatch.setattr(inputs,'verify',lambda s:calls.append('verify'))
    monkeypatch.setattr(inputs,'raw_inputs',lambda sha,sources:deepcopy(raw))
    def completed(root,sha,launch_sha,sources):
        calls.append(('completed',root,sha))
        if root==inputs.queue.budget.OUTPUT:return dict(report=dict(native_result_sha256='native')),{},{}
        assert root==inputs.queue.budget.native.OUTPUT and sha=='native'
        return {},dict(input_admission=dict(extended_queue_completion={'four':'original'})),{}
    monkeypatch.setattr(inputs,'completed',completed)
    monkeypatch.setattr(inputs,'verify_artifacts',lambda root,ids:calls.append(('artifacts',root,ids)))
    monkeypatch.setattr(inputs,'read_json',lambda root,name:dict(artifact_sha256={'launch.json':'launch'}))
    def queue_admit(four,sha,*,sources,full):
        assert four=={'four':'original'} and sha=='budget' and full is True
        calls.append('full_five_admission');return deepcopy(five)
    monkeypatch.setattr(inputs.queue,'admit',queue_admit)
    monkeypatch.setattr(inputs.queue,'verify_bound',lambda admission,sources:calls.append('verify_five'))
    return raw,five,calls


def test_admit_calls_actual_full_queue_path_and_preserves_model_and_negative_results(monkeypatch):
    raw,five,calls=fixture(monkeypatch)
    r=inputs.admit('raw','budget',{})
    assert calls[0]=='owners' and 'full_five_admission' in calls
    assert r['five_stage_queue_completion']==five and r['correction_admission']==raw['correction_admission']
    assert r['all_scientific_failures_retained'] and not r['queued_controller_changes_adopted']
    inputs.verify_bound(r,{})
    assert 'verify_five' in calls


@pytest.mark.parametrize('key,value',[('batch','different'),('hold','different')])
def test_mixed_batch_or_hold_identity_rejected(monkeypatch,key,value):
    raw,five,_=fixture(monkeypatch);five['budget_inputs'][key]=value
    with pytest.raises(ValueError,match='same original'):inputs.require_links(raw,five)


@pytest.mark.parametrize('flag',['all_scientific_failures_retained','complete_five_stage_queue_authenticated','budget_native_input_admission_fully_reexecuted'])
def test_queue_evidence_flags_must_be_complete_typed_booleans(monkeypatch,flag):
    raw,five,_=fixture(monkeypatch);five[flag]=1
    with pytest.raises(ValueError):inputs.require_links(raw,five)


@pytest.mark.parametrize('fault',['model','raw','budget','scope','success_filter'])
def test_bound_admission_cannot_change_inputs_or_scope(monkeypatch,fault):
    raw,five,calls=fixture(monkeypatch);r=inputs.admit('raw','budget',{})
    if fault=='model':r['correction_admission']['model']='different'
    if fault=='raw':r['raw_prefix_result_sha256']='different'
    if fault=='budget':r['budget_wait_result_sha256']='different'
    if fault=='scope':r['independent_study_policy_selected']=True
    if fault=='success_filter':r['all_scientific_failures_retained']=False
    with pytest.raises(ValueError):inputs.verify_bound(r,{})


def test_live_raw_owner_blocks_before_queue_or_artifact_access(monkeypatch):
    monkeypatch.setattr(inputs,'Path',lambda p:SimpleNamespace(read_text=lambda:inputs.BOOT))
    monkeypatch.setattr(inputs,'owner_live',lambda owner:True)
    monkeypatch.setattr(inputs.queue,'owners_ended',lambda:pytest.fail('queue touched before live raw rejection'))
    with pytest.raises(ValueError,match='raw replay is still live'):inputs.owners_ended()


def test_live_queue_blocks_before_raw_material_access(monkeypatch):
    monkeypatch.setattr(inputs,'Path',lambda p:SimpleNamespace(read_text=lambda:inputs.BOOT))
    monkeypatch.setattr(inputs,'owner_live',lambda owner:False)
    def waiting():raise ValueError('original budget live')
    monkeypatch.setattr(inputs.queue,'owners_ended',waiting)
    monkeypatch.setattr(inputs,'raw_inputs',lambda *a:pytest.fail('raw material touched while queue live'))
    with pytest.raises(ValueError,match='budget live'):inputs.admit('raw','budget',{})


def test_original_boot_is_required(monkeypatch):
    monkeypatch.setattr(inputs,'Path',lambda p:SimpleNamespace(read_text=lambda:'other'))
    with pytest.raises(ValueError,match='boot'):inputs.owners_ended()


@pytest.mark.parametrize('fault',[None,'raw_admission','model','native_prefix'])
def test_raw_material_calls_actual_completion_and_original_native_admission(monkeypatch,fault):
    calls=[];raw_launch=dict(input_admission={'actual':'admission'})
    native_launch=dict(model_state_sha256=inputs.replay.MODEL_SHA,input_admission=dict(
        prefix_result_sha256='prefix',adapter_batch_result_sha256='batch',correction_admission={'original':'model'}))
    native_result=dict(prospective_prefix_result_sha256='prefix')
    report=dict(model_state_sha256=inputs.replay.MODEL_SHA)
    if fault=='raw_admission':raw_launch['input_admission']={}
    if fault=='model':report['model_state_sha256']='wrong'
    if fault=='native_prefix':native_result['prospective_prefix_result_sha256']='wrong'
    monkeypatch.setattr(inputs,'owners_ended',lambda:None);monkeypatch.setattr(inputs,'verify',lambda s:None)
    def complete(root,sha,launch_sha,sources):
        if root==inputs.replay.OUTPUT:return {'raw':'result'},raw_launch,{'result.json':sha}
        return native_result,native_launch,{}
    monkeypatch.setattr(inputs,'completed',complete)
    def original_admit(sources):calls.append('original_admit');return {'actual':'admission'}
    def prefix_admit(root,result):
        assert root==inputs.replay.OUTPUT and result=={'raw':'result'}
        calls.append('prefix_admit');return deepcopy(report)
    monkeypatch.setattr(inputs.replay,'admit',original_admit)
    monkeypatch.setattr(inputs.prefix,'admit_prefix',prefix_admit)
    if fault:
        with pytest.raises(ValueError):inputs.raw_inputs('raw',{})
    else:
        r=inputs.raw_inputs('raw',{})
        assert calls==['original_admit','prefix_admit'] and r['prefix_report']==report
        assert r['original_batch_result_sha256']=='batch' and r['correction_admission']=={'original':'model'}
