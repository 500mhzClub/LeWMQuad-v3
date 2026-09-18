"""No truncation, changed prehistory, or incomplete intervention can qualify."""
from copy import deepcopy
import numpy as np
import pytest
from scripts import direct_flow_commitment_contact_native_prefix_development as prefix
from lewm.tests.test_contact_anchored_direct_flow_controller_prefix_development import early, boundary


def fixture():
    originals=[];saved=[];observed=[];tape=[]
    for frame in range(prefix.FRAMES):
        old,new = boundary() if frame==prefix.INTERVENTION else early()
        if frame < prefix.INTERVENTION:
            old['tick']=new['tick']=frame
            if frame < 3: old['new_selection']=new['new_selection']=None
        old['selected_action']=new['selected_action']=None
        check=prefix.replay.compare(old,new,old['requested_command'],new['original_visual_evidence'],frame=frame)
        originals.append(dict(tick=frame,observation_index=frame,pre_sample_index=749+50*frame,decision=old))
        saved.append(dict(tick=frame,decision=new,comparison=check,original_requested_command=old['requested_command'],
            complete_original_decision_reconstructed=True,public_input_arrays_unchanged=True,public_input_sha256='packet_'+str(frame)))
        observed.append(dict(tick=frame,candidate=deepcopy(new['original_visual_evidence']),public_packet_sha256='packet_'+str(frame)))
        tape.append(dict(tick=frame,completed=True,pre_sample_index=749+50*frame,post_sample_index=799+50*frame,
            requested_command=deepcopy(old['requested_command'])))
    report=dict(complete_original_controller_decisions_reconstructed=True,two_fresh_original_model_copies=True,frames=562,exact_candidate_original_decisions=561,original_forecasts_compared=558,
        boundary_terminal=None,boundary_failure=None,model_state_sha256=prefix.replay.MODEL_SHA,
        model_state_unchanged=True,following_recorded_observations_consumed=False,
        actual_original_commands_before_intervention_exact=True,new_command_executed=False,
        original_failed_outcome_preserved=True,native_execution=False,navigation_qualified=False,goal_achieved=False,
        boundary_comparison=deepcopy(saved[-1]['comparison']),boundary_requested_command=[0.,0.,0.],boundary_selected_action=None)
    return report,originals,saved,tape,observed


def test_complete_controller_prefix_reconstructs_and_allows_an_honest_hold():
    report,old,saved,tape,visual=fixture()
    r=prefix.reconstruct(report,iter(old),iter(saved),tape,iter(visual))
    assert r==dict(frames=562,first_intervention_frame=561,original_forecasts_compared=558,complete_saved_comparisons_reconstructed=True)
    prefix.executed_boundary([tape,deepcopy(tape)],report)


@pytest.mark.parametrize('fault',['short','extra','forecast','observer','packet','comparison','command',
    'incomplete','endpoint','input_mutation','order','summary_action','summary_command'])
def test_invalid_saved_prefix_is_rejected(fault):
    r,old,saved,tape,visual=fixture()
    if fault=='short':saved.pop()
    elif fault=='extra':saved.append(deepcopy(saved[-1]))
    elif fault=='forecast':saved[500]['decision']['new_selection']['prediction']['x'][0]+=1
    elif fault=='observer':saved[-1]['decision']['original_visual_evidence']['status']='WRONG'
    elif fault=='packet':saved[500]['public_input_sha256']='wrong'
    elif fault=='comparison':saved[500]['comparison']['stop']=True
    elif fault=='command':tape[500]['requested_command']=[.2,0.,0.]
    elif fault=='incomplete':tape[-1]['completed']=False
    elif fault=='endpoint':tape[-1]['post_sample_index']-=1
    elif fault=='input_mutation':saved[-1]['public_input_arrays_unchanged']=False
    elif fault=='order':old[500]['observation_index']-=1
    elif fault=='summary_action':r['boundary_selected_action']='forward'
    else:r['boundary_requested_command']=[.2,0.,0.]
    with pytest.raises(ValueError):prefix.reconstruct(r,old,saved,tape,visual)


@pytest.mark.parametrize('key,value',[
    ('frames',True),('frames',561),('exact_candidate_original_decisions',560),('original_forecasts_compared',True),
    ('boundary_terminal','SENSOR_OR_MODEL_FAILURE'),('boundary_failure','floor failed'),
    ('model_state_sha256','wrong'),('following_recorded_observations_consumed',True),
    ('new_command_executed',True),('boundary_requested_command',[float('nan'),0.,0.]),
    ('boundary_requested_command',[False,0.,0.])])
def test_incomplete_negative_or_mislabeled_report_is_rejected(key,value):
    report,*_=fixture();report[key]=value
    with pytest.raises(ValueError):prefix.boundary(report)


@pytest.mark.parametrize('fault',['incomplete','bool_tick','pre','post','short','earlier','wrong_boundary'])
def test_candidate_command_must_actually_complete(fault):
    report,_,_,old,_=fixture();new=deepcopy(old)
    if fault=='incomplete':new[-1]['completed']=False
    elif fault=='bool_tick':new[0]['tick']=False
    elif fault=='pre':new[-1]['pre_sample_index']-=1
    elif fault=='post':new[-1]['post_sample_index']-=1
    elif fault=='short':new.pop()
    elif fault=='earlier':new[550]['requested_command']=[0.,0.,.45]
    else:new[-1]['requested_command']=[.2,0.,0.]
    with pytest.raises(ValueError):prefix.executed_boundary([old,new],report)


def physical_fixture(monkeypatch,tmp_path,fault=None):
    report,old,saved,tape,visual=fixture()
    prior=tmp_path/'prior';current=tmp_path/'current';root=tmp_path/'prefix'
    prior.mkdir();current.mkdir()
    actual=deepcopy(old)
    for a,s in zip(actual,saved,strict=True):a['decision']=deepcopy(s['decision'])
    new_tape=deepcopy(tape)
    values=np.arange(prefix.PHYSICS_SAMPLES+50,dtype=np.float64);other=values.copy()
    other[prefix.PHYSICS_SAMPLES:]+=100  # Later physics must not be required equal.
    if fault=='physics':other[prefix.PHYSICS_SAMPLES-1]+=1
    if fault=='truncated_physics':other=other[:prefix.PHYSICS_SAMPLES-1]
    if fault=='missing_boundary_samples':other=other[:prefix.PHYSICS_SAMPLES]
    if fault=='partial_boundary_samples':other=other[:prefix.PHYSICS_SAMPLES+49]
    np.savez(prior/'physics_trace.npz',samples=values)
    np.savez(current/'physics_trace.npz',samples=other)
    if fault=='decision':actual[-1]['decision']['evidence']={'wrong':'pose'}
    if fault=='completion':new_tape[-1]['completed']=False
    if fault=='row_clock':actual[-1]['pre_sample_index']-=1
    if fault=='short':actual.pop()
    monkeypatch.setattr(prefix,'artifact_path',lambda parent,name:parent/name)
    def rows(p):
        source=old if p==prior else actual if p==current else saved if p==root else visual
        yield from deepcopy(source)
    monkeypatch.setattr(prefix,'read_rows',rows)
    monkeypatch.setattr(prefix,'read_json',lambda p,n:deepcopy(tape if p==prior else new_tape))
    def packets(p):
        for i in range(prefix.FRAMES):
            yield 'wrong' if fault=='public' and p==current and i==561 else 'packet_'+str(i)
    monkeypatch.setattr(prefix,'public_packets',packets)
    return prior,current,root,report


def test_exact_28800_samples_and_completed_hold_are_checked(monkeypatch,tmp_path):
    result=prefix.compare(*physical_fixture(monkeypatch,tmp_path))
    assert result['physical_prefix_samples']==28800 and result['common_prefix_frames']==562
    assert result['candidate_intervention_command_completed']
    assert not result['intervention_command_changed']
    assert not result['following_physical_outcomes_compared'] and not result['navigation_verified']


@pytest.mark.parametrize('fault',['physics','truncated_physics','missing_boundary_samples','partial_boundary_samples',
    'decision','completion','row_clock','short','public'])
def test_changed_native_prefix_cannot_qualify(monkeypatch,tmp_path,fault):
    with pytest.raises(ValueError):prefix.compare(*physical_fixture(monkeypatch,tmp_path,fault))


def admission_fixture(monkeypatch,tmp_path,fault=None):
    report,old,saved,tape,visual=fixture()
    root=tmp_path/'prefix';root.mkdir()
    sources={'source.py':'a'*64};original_inputs={'old.json':'b'*64}
    observer_ids={'result.json':prefix.replay.OBSERVER_SHA,'launch.json':'c'*64}
    artifacts={n:'d'*64 for n in ('launch.json','context_decisions.jsonl.gz','report.json')}
    artifacts['launch.json']=prefix.LAUNCH_SHA
    result=dict(status='CONTACT_ANCHORED_DIRECT_FLOW_CONTROLLER_PREFIX_V1_COMPLETE',native_execution=False,
        source_sha256=sources,artifact_sha256=artifacts,report=report)
    launch=dict(source_sha256=sources,case=list(prefix.replay.CASE),model_state_sha256=prefix.replay.MODEL_SHA,
        boundary_frame=561,implementation_class='DirectFlowCommitmentContactController',
        original_completed_worker_admitted=True,two_fresh_original_model_copies=True,actual_assigned_model_loader_required=True,full_training_ancestry_reexecuted=False,observer_artifact_sha256=observer_ids,input_artifact_sha256=original_inputs)
    caller_sources=deepcopy(sources)
    if fault=='terminal_failure':(root/'failure.json').write_text('{}')
    elif fault=='status':result['status']='INCOMPLETE'
    elif fault=='native_claim':result['native_execution']=True
    elif fault=='missing_artifact':artifacts.pop('report.json')
    elif fault=='changed_source':caller_sources['source.py']='0'*64
    elif fault=='case':launch['case'][0]='wrong'
    elif fault=='model':launch['model_state_sha256']='0'*64
    elif fault=='admission':launch['original_completed_worker_admitted']=False
    elif fault=='observer_identity':observer_ids['result.json']='0'*64
    elif fault=='input_subset':launch['input_artifact_sha256']={}
    elif fault=='negative_controller':report['boundary_comparison']['full_controller_recovered']=False
    elif fault=='raw_packet':saved[0]['public_input_sha256']=visual[0]['public_packet_sha256']='wrong'
    checked=[]
    monkeypatch.setattr(prefix.replay,'owners_ended',lambda:None)
    monkeypatch.setattr(prefix.replay,'owner_live',lambda owner:False)
    monkeypatch.setattr(prefix.replay,'verify_inputs',lambda *a:deepcopy(original_inputs))
    monkeypatch.setattr(prefix.replay,'OUTPUT',root)
    monkeypatch.setattr(prefix,'verify',lambda bindings:checked.append(('sources',dict(bindings))))
    monkeypatch.setattr(prefix,'verify_artifacts',lambda p,b:checked.append((p,dict(b))))
    prior=prefix.replay.native.OUTPUT/prefix.replay.CASE[0]
    def read(p,name):
        if p==root:return {'result.json':result,'launch.json':launch,'report.json':report}[name]
        if p==prefix.replay.observer.OUTPUT:return {'input_artifact_sha256':original_inputs}
        if p==prior and name=='command_tape.json':return tape
        raise AssertionError((p,name))
    monkeypatch.setattr(prefix,'read_json',read)
    def rows(p):yield from deepcopy(saved if p==root else old if p==prior else visual)
    monkeypatch.setattr(prefix,'read_rows',rows)
    monkeypatch.setattr(prefix,'public_packets',lambda p:('packet_'+str(i) for i in range(prefix.FRAMES)))
    return caller_sources,report,checked,root,original_inputs


def test_admission_reconstructs_every_row_and_requests_all_bindings(monkeypatch,tmp_path):
    sources,report,checked,root,original_inputs=admission_fixture(monkeypatch,tmp_path)
    assert prefix.admit_prefix('e'*64,sources)==report
    assert (root,{'result.json':'e'*64,'launch.json':prefix.LAUNCH_SHA}) in checked
    assert (prefix.replay.native.OUTPUT,original_inputs) in checked
    assert ('sources',sources) in checked


@pytest.mark.parametrize('fault',['terminal_failure','status','native_claim','missing_artifact','changed_source',
    'case','model','admission','observer_identity','input_subset','negative_controller','raw_packet'])
def test_admission_rejects_unqualified_or_changed_evidence(monkeypatch,tmp_path,fault):
    sources,*_=admission_fixture(monkeypatch,tmp_path,fault)
    with pytest.raises(ValueError):prefix.admit_prefix('e'*64,sources)


@pytest.mark.parametrize('action',['hold','left_turn','forward'])
def test_selected_commands_and_change_accounting_reconstruct(action):
    report,old,saved,tape,visual=fixture();new=saved[-1]['decision']
    command=list(prefix.replay.candidate_commands(action)[0])
    new['new_selection']['action']=new['selected_action']=action
    new['requested_command']=command
    saved[-1]['comparison']=prefix.replay.compare(old[-1]['decision'],new,[0.,0.,0.],visual[-1]['candidate'],frame=prefix.INTERVENTION)
    report.update(boundary_requested_command=command,boundary_selected_action=action,
        boundary_comparison=deepcopy(saved[-1]['comparison']))
    prefix.reconstruct(report,old,saved,tape,visual)
    actual=deepcopy(tape);actual[-1]['requested_command']=command
    prefix.executed_boundary([tape,actual],report)
    assert report['boundary_comparison']['requested_command_changed'] is (action!='hold')


def test_live_replay_cannot_be_admitted_from_partial_outputs(monkeypatch,tmp_path):
    sources,*_=admission_fixture(monkeypatch,tmp_path)
    monkeypatch.setattr(prefix.replay,'owner_live',lambda owner:True)
    monkeypatch.setattr(prefix,'verify_artifacts',lambda *a:pytest.fail('must not open partial result'))
    with pytest.raises(ValueError,match='still live'):prefix.admit_prefix('e'*64,sources)
