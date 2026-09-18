"""Keep the full ordered queue, negative outcomes and actual verifier calls."""
from copy import deepcopy
import pytest
from scripts import independent_round_trip_later_diagnostics_evidence_development as join


def fixtures(stage,success=0):
    w=stage.waiter
    launch=dict(boot_id=join.BOOT,waiter_pid=stage.owner['pid'],
        original_owners={s[0]:s[1] for s in w.PREREQUISITES},
        prerequisite_launch_sha256={s[0]:s[3] for s in w.PREREQUISITES},
        planned_case=list(w.native.CASE),maximum_wait_s=w.WAIT_SECONDS,
        automatic_retry=False,source_changes_permitted=False,
        native_workers_while_waiting=0,native_workers_after_original_completion=1)
    completion=dict(native_result_sha256='f'*64,measured_round_trip_successes=success,
        complete_native_worker_and_artifact_roster_verified=True,actual_physical_prefix_reconstructed=True,
        scientific_success_required=False,final_independent_population_policy_review_performed=False)
    result=dict(status=stage.status,automatic_retry=False,
        artifact_sha256={n:stage.launch_sha256 if n=='launch.json' else 'a'*64 for n in join.WAIT_FILES},
        report=deepcopy(completion),navigation_qualified=False,real_time_qualified=False,
        hardware_qualified=False,goal_achieved=False)
    inputs=join.expected_inputs(stage,'b'*64)
    return result,launch,deepcopy(inputs),completion,inputs


@pytest.mark.parametrize('stage',join.STAGES,ids=lambda s:s.name)
@pytest.mark.parametrize('success',[0,1])
def test_both_negative_and_positive_completed_outcomes_are_admissible(stage,success):
    join.require_waiter(stage,*fixtures(stage,success))


@pytest.mark.parametrize('field,value',[
    ('waiter_pid',0),('boot_id','other'),('original_owners',{}),('prerequisite_launch_sha256',{}),
    ('planned_case',[]),('maximum_wait_s',0),('automatic_retry',True),('source_changes_permitted',True),
    ('native_workers_while_waiting',False),('native_workers_after_original_completion',True),
])
def test_waiter_owner_and_definition_changes_rejected(field,value):
    stage=join.STAGES[0];args=fixtures(stage);args[1][field]=value
    with pytest.raises(ValueError):join.require_waiter(stage,*args)


@pytest.mark.parametrize('fault',['status','artifact','raw','predecessor','report','scope','success_bool',
                                  'prefix_flag','success_required','review_done'])
def test_missing_links_incomplete_evidence_or_success_selection_rejected(fault):
    stage=join.STAGES[1];result,launch,receipt,completion,inputs=fixtures(stage)
    if fault=='status':result['status']='partial'
    if fault=='artifact':result['artifact_sha256'].pop('events.jsonl')
    if fault=='raw':receipt['raw']='c'*64
    if fault=='predecessor':receipt[stage.predecessor_key]='c'*64
    if fault=='report':result['report']['native_result_sha256']='c'*64
    if fault=='scope':result['goal_achieved']=True
    if fault=='success_bool':completion['measured_round_trip_successes']=False
    if fault=='prefix_flag':completion['actual_physical_prefix_reconstructed']=1
    if fault=='success_required':completion['scientific_success_required']=True
    if fault=='review_done':completion['final_independent_population_policy_review_performed']=True
    if fault in ('success_bool','prefix_flag','success_required','review_done'):
        result['report']=deepcopy(completion)
    with pytest.raises(ValueError):join.require_waiter(stage,result,launch,receipt,completion,inputs)


@pytest.mark.parametrize('live_stage',join.STAGES,ids=lambda s:s.name)
def test_live_later_owner_rejected_before_any_artifact_access(monkeypatch,live_stage):
    monkeypatch.setattr(join,'owner_live',lambda owner:owner==live_stage.owner)
    def forbidden(*a,**k):raise AssertionError('no evidence reads or older admission while later owner is live')
    monkeypatch.setattr(join,'verify',forbidden)
    monkeypatch.setattr(join,'completed',forbidden)
    monkeypatch.setattr(join.previous,'owners_ended',forbidden)
    with pytest.raises(ValueError,match='still live'):
        join.admit({}, {s.name:'a'*64 for s in join.STAGES},sources={})


def fake_admission(monkeypatch):
    calls=[]
    monkeypatch.setattr(join,'owners_ended',lambda:calls.append('owners_ended'))
    monkeypatch.setattr(join,'verify',lambda sources:None)
    monkeypatch.setattr(join.previous,'verify_bound',lambda prior,sources:calls.append(('five',deepcopy(prior))))
    def stage(spec,sha,preceding,sources):
        calls.append((spec.name,sha,preceding))
        return dict(stage=spec.name,waiter_result_sha256=sha,prerequisite_result_sha256=join.expected_inputs(spec,preceding),
            measured_round_trip_successes=0,readout={'failure':'retained'})
    monkeypatch.setattr(join,'admit_stage',stage)
    five=dict(budget_wait_result_sha256='b'*64)
    ids={s.name:str(i+1)*64 for i,s in enumerate(join.STAGES)}
    return five,ids,calls


def test_all_three_ordered_links_and_negative_outcomes_retained_without_policy_selection(monkeypatch):
    five,ids,calls=fake_admission(monkeypatch)
    result=join.admit(five,ids,sources={})
    assert calls==['owners_ended',('five',five),('sustained_turn','1'*64,'b'*64),
        ('contact_flow','2'*64,'1'*64),('chained_anchor','3'*64,'2'*64),'owners_ended']
    assert [r['stage'] for r in result['later_diagnostics']]==[s.name for s in join.STAGES]
    assert all(r['measured_round_trip_successes']==0 for r in result['later_diagnostics'])
    assert result['all_scientific_failures_retained'] is True
    assert result['final_policy_review_completed'] is False
    assert result['population_execution_permitted'] is False
    assert result['population_definition_selected'] is False
    result['original_five_stage_admission']['changed']=True
    assert 'changed' not in five


@pytest.mark.parametrize('fault',['missing','extra','invalid_sha'])
def test_incomplete_or_extra_diagnostic_roster_rejected(monkeypatch,fault):
    five,ids,_=fake_admission(monkeypatch)
    if fault=='missing':ids.pop('contact_flow')
    if fault=='extra':ids['alternate']='a'*64
    if fault=='invalid_sha':ids['chained_anchor']='pending'
    with pytest.raises(ValueError):join.admit(five,ids,sources={})


def test_whole_bound_evidence_reconstructed_and_review_stays_uncompleted(monkeypatch):
    five,ids,_=fake_admission(monkeypatch);result=join.admit(five,ids,sources={})
    join.verify_bound(result,{})
    monkeypatch.setattr(join.final,'review_evidence',lambda inputs,prior:dict(original_five='retained'))
    review=join.review_evidence({},result,sources={})
    assert review['diagnostic_count']==8 and review['final_policy_review_completed'] is False
    changed=deepcopy(result);changed['later_diagnostics'][1]['readout']['failure']='erased'
    with pytest.raises(ValueError,match='whole eight'):join.verify_bound(changed,{})


@pytest.mark.parametrize('stage',join.STAGES,ids=lambda s:s.name)
@pytest.mark.parametrize('fault',[None,'authenticator','readout','prefix','boolean_outcome'])
def test_original_native_authenticator_runs_and_saved_worker_must_match(monkeypatch,stage,fault):
    result,launch,receipt,completion,inputs=fixtures(stage)
    name=stage.waiter.native.CASE[0];calls=[]
    worker=dict(verified_round_trip=False,collection={'terminal':'negative'},
        readout={'failure':'retained'},prefix_comparison={'exact':False})
    saved=deepcopy(worker)
    if fault=='readout':saved['readout']['failure']='rewritten'
    if fault=='prefix':saved['prefix_comparison']['exact']=True
    if fault=='boolean_outcome':worker['verified_round_trip']=0
    native_result=dict(artifact_sha256={'launch.json':'d'*64},conditions=[worker])
    def completed(root,*args):
        return (result,launch,{'result.json':'e'*64}) if root==stage.waiter.OUTPUT else (native_result,{}, {'result.json':'f'*64})
    def read(root,path):
        if root==stage.waiter.OUTPUT:
            return {'input_completion.json':receipt,'native_completion.json':completion}[path]
        return {'result.json':native_result,name+'_worker_terminal.json':worker,
            name+'_readout.json':saved['readout'],name+'_prefix_comparison.json':saved['prefix_comparison']}[path]
    def authenticate(sources,actual_inputs):
        calls.append(deepcopy(actual_inputs));actual=deepcopy(completion)
        if fault=='authenticator':actual['actual_physical_prefix_reconstructed']=False
        return actual
    monkeypatch.setattr(join,'completed',completed);monkeypatch.setattr(join,'read_json',read)
    monkeypatch.setattr(join,'verify_artifacts',lambda *args:None)
    monkeypatch.setattr(stage.waiter,'authenticate_completed',authenticate)
    if fault:
        with pytest.raises(ValueError):join.admit_stage(stage,'e'*64,'b'*64,{})
    else:
        actual=join.admit_stage(stage,'e'*64,'b'*64,{})
        assert actual['readout']==worker['readout'] and actual['measured_round_trip_successes']==0
    assert calls==[inputs]
