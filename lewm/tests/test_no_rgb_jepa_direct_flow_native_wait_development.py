"""No native handoff before exact original completion, and no automatic retry."""
from copy import deepcopy
import json
import numpy as np
import pytest
from scripts import await_go2_no_rgb_jepa_direct_flow_maze02_native_v1 as waiter
from lewm.tests.test_no_rgb_jepa_direct_flow_native_prefix_development import fixture as prefix_fixture


def identities():return {s[0]:str(i+1)*64 for i,s in enumerate(waiter.PREREQUISITES)}


def test_every_original_owner_must_end_before_handoff(monkeypatch):
    rounds=iter([True]*5+[False,True,True,True,True]+[False]*5);sleeps=[];events=[]
    ids=identities()
    monkeypatch.setattr(waiter,'owner_live',lambda owner:next(rounds))
    monkeypatch.setattr(waiter,'completion_identity',lambda spec,sources:ids[spec[0]])
    monkeypatch.setattr(waiter,'verify',lambda sources:None)
    assert waiter.wait_for_inputs({},lambda *a,**k:events.append(k),sleep=sleeps.append,clock=lambda:0)==ids
    assert sleeps==[30,30] and len(events)==2
    assert events[1]['completed_result_sha256']=={'prefix':ids['prefix']}


def test_first_completed_identity_cannot_change(monkeypatch):
    rounds=iter([False,True,True,True,True]+[False]*5);hashes=iter(['first','changed'])
    monkeypatch.setattr(waiter,'owner_live',lambda owner:next(rounds))
    monkeypatch.setattr(waiter,'completion_identity',lambda *a:next(hashes))
    with pytest.raises(ValueError,match='identity changed'):
        waiter.wait_for_inputs({},lambda *a,**k:None,sleep=lambda n:None,clock=lambda:0)


def test_missing_completion_does_not_restart(monkeypatch):
    monkeypatch.setattr(waiter,'owner_live',lambda owner:False)
    def missing(*a):raise ValueError('original owner ended without completion')
    monkeypatch.setattr(waiter,'completion_identity',missing)
    with pytest.raises(ValueError,match='without completion'):
        waiter.wait_for_inputs({},lambda *a,**k:None,sleep=lambda n:pytest.fail('no sleep'),clock=lambda:0)


def test_owner_identity_failure_is_not_completion(monkeypatch):
    def reused(owner):raise ValueError('PID identity changed')
    monkeypatch.setattr(waiter,'owner_live',reused)
    monkeypatch.setattr(waiter,'completion_identity',lambda *a:pytest.fail('not complete'))
    with pytest.raises(ValueError,match='identity changed'):waiter.wait_for_inputs({},lambda *a,**k:None,clock=lambda:0)


def test_bounded_wait_has_no_replacement(monkeypatch):
    times=iter([0,waiter.WAIT_SECONDS]);monkeypatch.setattr(waiter,'owner_live',lambda owner:True)
    with pytest.raises(ValueError,match='expired'):
        waiter.wait_for_inputs({},lambda *a,**k:None,sleep=lambda n:pytest.fail('no sleep'),clock=lambda:next(times))


def test_command_binds_all_five_exact_results():
    ids=identities();command=waiter.command_for(ids)
    assert command[1]==waiter.native.SOURCE
    assert dict(zip(command[2::2],command[3::2],strict=True))=={
        '--controller-prefix-result-sha256':ids['prefix'],'--adapter-batch-result-sha256':ids['batch'],
        '--frontier-wait-result-sha256':ids['frontier'],'--hold-wait-result-sha256':ids['hold'],
        '--contact-wait-result-sha256':ids['contact']}


@pytest.mark.parametrize('fault',['missing','extra','invalid_hash','wrong_type'])
def test_incomplete_command_identities_rejected(fault):
    ids=identities()
    if fault=='missing':ids.pop('contact')
    elif fault=='extra':ids['other']='f'*64
    elif fault=='invalid_hash':ids['prefix']='not-a-result'
    else:ids['batch']=None
    with pytest.raises(ValueError):waiter.command_for(ids)


def completion_fixture(monkeypatch,tmp_path,fault=None,key='prefix'):
    report=prefix_fixture()[0];report['boundary_requested_command']=[0.,0.,-.45]
    report['boundary_selected_action']='right_turn';report['boundary_comparison']['requested_command_changed']=True
    root=tmp_path/key;root.mkdir();(root/'result.json').write_text('{}')
    sources={'source':'s'};launch={'source_sha256':sources}
    result=dict(status='COMPLETE',source_sha256=deepcopy(sources),artifact_sha256=deepcopy(waiter.PREFIX_BINDINGS),report=report,
        all_fixed_cases_executed=True,conditions=[{}]*6,automatic_retry=False)
    if fault=='failure':(root/'failure.json').write_text('{}')
    elif fault=='missing':(root/'result.json').unlink()
    elif fault=='status':result['status']='FAILED'
    elif fault=='source':result['source_sha256']['source']='changed'
    elif fault=='artifact':result['artifact_sha256']['report.json']='wrong'
    elif fault=='negative':report['boundary_comparison']['full_controller_recovered']=False
    elif fault=='command':report['boundary_requested_command']=[0.,0.,0.]
    elif fault=='batch_count':result['conditions'].pop()
    elif fault=='retry':result['automatic_retry']=True
    checked=[]
    monkeypatch.setattr(waiter,'verify_artifacts',lambda root,ids:checked.append(dict(ids)))
    monkeypatch.setattr(waiter,'read_json',lambda root,name:deepcopy(result if name=='result.json' else launch if name=='launch.json' else report))
    spec=(key,{},root,waiter.PREFIX_BINDINGS['launch.json'],'COMPLETE')
    return spec,sources,checked


def test_light_completion_checks_do_not_hash_the_recorded_dataset(monkeypatch,tmp_path):
    spec,sources,checked=completion_fixture(monkeypatch,tmp_path)
    assert len(waiter.completion_identity(spec,sources))==64
    assert len(checked)==2
    assert set(checked[0])=={'launch.json','result.json'} and set(checked[1])=={'report.json'}


@pytest.mark.parametrize('fault',['failure','missing','status','source','artifact','negative','command'])
def test_changed_or_negative_prefix_completion_rejected(monkeypatch,tmp_path,fault):
    spec,sources,_=completion_fixture(monkeypatch,tmp_path,fault)
    with pytest.raises(ValueError):waiter.completion_identity(spec,sources)


@pytest.mark.parametrize('key,fault',[('batch','batch_count'),('frontier','retry')])
def test_incomplete_batch_or_retried_queue_is_not_completion(monkeypatch,tmp_path,key,fault):
    spec,sources,_=completion_fixture(monkeypatch,tmp_path,fault,key)
    with pytest.raises(ValueError):waiter.completion_identity(spec,sources)


def test_failed_child_is_started_once_and_preserved_without_retry(monkeypatch,tmp_path):
    root=tmp_path/'wait';child_root=tmp_path/'native';ids=identities();calls=[]
    monkeypatch.setattr(waiter,'OUTPUT',root);monkeypatch.setattr(waiter.native,'OUTPUT',child_root)
    monkeypatch.setattr(waiter,'validate_root',lambda *a,**k:None)
    monkeypatch.setattr(waiter,'create_output',lambda p:p.mkdir())
    monkeypatch.setattr(waiter,'prepared_sources',lambda:{})
    monkeypatch.setattr(waiter,'owner_live',lambda owner:True)
    monkeypatch.setattr(waiter,'wait_for_inputs',lambda *a:ids)
    monkeypatch.setattr(waiter.native.inputs,'owners_ended',lambda:None)
    monkeypatch.setattr(waiter,'verify',lambda *a:None)
    monkeypatch.setattr(waiter.native,'hardware',lambda:dict(memory_available_bytes=64*1024**3,artifact_free_bytes=100*1024**3))
    monkeypatch.setattr(waiter.native,'require_native_idle',lambda:None)
    monkeypatch.setattr(waiter.time,'sleep',lambda n:None)
    monkeypatch.setattr(waiter,'authenticate_completed',lambda *a:pytest.fail('failed child is not completion'))
    class Child:
        pid=123;returncode=7
        def __init__(self):self.polls=iter([None,7])
        def poll(self):return next(self.polls)
    def popen(command,**kwargs):
        calls.append(command);kwargs['stdout'].write(b'synthetic child failure\n');return Child()
    monkeypatch.setattr(waiter.subprocess,'Popen',popen)
    with pytest.raises(ValueError,match='child failed'):waiter.main()
    assert calls==[waiter.command_for(ids)]
    assert (root/'native_stdout.log').read_text()=='synthetic child failure\n'
    failure=json.loads((root/'failure.json').read_text())
    assert failure['automatic_retry'] is False and failure['original_work_retained'] is True
    assert not child_root.exists()


def native_completion_fixture(monkeypatch,tmp_path,fault=None):
    from lewm.tests.test_no_rgb_jepa_direct_flow_native_launcher_development import fixture
    record,audit,prefix=fixture();ids=identities();root=tmp_path/'native';root.mkdir()
    name=waiter.native.CASE[0];case_root=root/name;case_root.mkdir()
    (root/'result.json').write_text('{}');np.savez(case_root/'physics_trace.npz',physics_contact=np.zeros(2))
    prefix.update(boundary_requested_command=[0.,0.,-.45],boundary_selected_action='right_turn')
    prefix['boundary_comparison']['requested_command_changed']=True
    record['prefix_comparison'].update(candidate_intervention_command=[0.,0.,-.45],intervention_command_changed=True)
    record.update(readout={'round_trip':False},worker_log_sha256='b'*64,
        artifact_sha256={name+'/result.json':'a'*64,name+'/physics_trace.npz':'a'*64})
    bindings=record['artifact_sha256']|{name+s:'a'*64 for s in ('_worker_terminal.json','_audit.json','_prefix_comparison.json','_readout.json')}
    bindings.update({'launch.json':'a'*64,name+'_worker.log':'b'*64})
    sources={'source':'s'}
    launch=dict(source_sha256=sources,input_admission={'prefix_report':prefix})
    result=dict(status='NO_RGB_JEPA_DIRECT_FLOW_MAZE02_PILOT_V1_COMPLETE',source_sha256=deepcopy(sources),
        artifact_sha256=bindings,prospective_prefix_result_sha256=ids['prefix'],original_adapter_batch_result_sha256=ids['batch'],
        original_ordered_waiter_result_sha256={k:ids[k] for k in ('frontier','hold','contact')},conditions=[record],measured_round_trip_successes=0)
    if fault=='link':result['prospective_prefix_result_sha256']='0'*64
    elif fault=='source':result['source_sha256']['source']='changed'
    elif fault=='missing_artifact':bindings.pop(name+'_audit.json')
    elif fault=='worker_log':record['worker_log_sha256']='f'*64
    elif fault=='false_success':
        record['verified_round_trip']=audit['verified_round_trip']=True;result['measured_round_trip_successes']=1
    elif fault=='collection':record['collection']['direct_corner_flow_missingness_fallback_enabled']=False
    saved_worker=deepcopy(record)
    if fault=='worker':saved_worker['unexpected']='changed'
    records={'result.json':result,'launch.json':launch,name+'_worker_terminal.json':saved_worker,
        name+'/result.json':record['collection'],name+'_prefix_comparison.json':record['prefix_comparison'],
        name+'_readout.json':record['readout'],name+'_audit.json':audit}
    monkeypatch.setattr(waiter.native,'OUTPUT',root)
    monkeypatch.setattr(waiter,'read_json',lambda p,n:deepcopy(records[n]))
    monkeypatch.setattr(waiter,'verify',lambda *a:None)
    monkeypatch.setattr(waiter,'verify_artifacts',lambda *a:None)
    monkeypatch.setattr(waiter.native,'verify_inputs',lambda *a:None)
    monkeypatch.setattr(waiter.native,'artifacts',lambda *a:['result.json','physics_trace.npz'])
    monkeypatch.setattr(waiter.native,'case_readout',lambda *a:{'wrong':True} if fault=='readout' else {'round_trip':False})
    return sources,ids


def test_complete_audited_scientific_failure_is_valid_handoff_completion(monkeypatch,tmp_path):
    result=waiter.authenticate_completed(*native_completion_fixture(monkeypatch,tmp_path))
    assert result['measured_round_trip_successes']==0 and result['scientific_success_required'] is False
    assert result['complete_native_worker_and_artifact_roster_verified'] is True


@pytest.mark.parametrize('fault',['link','source','missing_artifact','worker','worker_log','false_success','collection','readout'])
def test_incomplete_or_changed_native_completion_rejected(monkeypatch,tmp_path,fault):
    with pytest.raises(ValueError):waiter.authenticate_completed(*native_completion_fixture(monkeypatch,tmp_path,fault))
