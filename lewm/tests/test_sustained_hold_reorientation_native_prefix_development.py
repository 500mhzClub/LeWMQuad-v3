"""Synthetic causal and physical prefix checks; no real native outcome claimed."""
from copy import deepcopy
import numpy as np
import pytest
from lewm.tests.test_sustained_hold_reorientation_raw_prefix_development import pair
from lewm.sustained_hold_reorientation_prefix_development import compare_step,identity
from scripts import sustained_hold_reorientation_native_prefix_development as prefix


def fixture():
    originals=[];prospective=[];tape=[];expected=[]
    for i in range(407):
        old,new=pair(i)
        row=dict(tick=i,observation_index=i,pre_sample_index=749+50*i,decision=old)
        check=compare_step(old,new,old['requested_command'],frame=i,expected_selection_sha256=identity(new['new_selection']))
        originals.append(row)
        prospective.append(dict(tick=i,decision=new,comparison=check,
            original_requested_command=old['requested_command'],original_complete_decision_reconstructed=True,
            public_input_arrays_unchanged=True,complete_retained_contact_state_equal=True,public_input_sha256='packet_'+str(i)))
        tape.append(dict(tick=i,completed=True,requested_command=old['requested_command'],
            pre_sample_index=749+50*i,post_sample_index=799+50*i))
        expected.append(dict(original_row_sha256=prefix.replay.saved.identity(row),candidate_selection_sha256=identity(new['new_selection'])))
    report=dict(frames=407,first_changed_command_frame=406,raw_model_forecast_comparisons=404,
        original_requested_command=[0.,0.,0.],candidate_requested_command=[0.,0.,.45])
    return report,originals,prospective,tape,expected


def test_complete_raw_comparisons_reconstruct_both_metadata_and_request_boundaries():
    r=prefix.reconstruct(*fixture())
    assert r['frames']==407 and r['first_intervention_frame']==406 and r['raw_model_forecast_comparisons']==404


@pytest.mark.parametrize('fault',['truncated','extra','counter','boundary','command','comparison','incomplete','observed_state','forecast','expected','original_row','receipt'])
def test_altered_raw_evidence_or_summary_is_rejected(fault):
    r,old,saved,tape,expected=fixture()
    if fault=='truncated':saved.pop()
    if fault=='extra':saved.append(deepcopy(saved[-1]))
    if fault=='counter':r['raw_model_forecast_comparisons']-=1
    if fault=='boundary':r['first_changed_command_frame']-=1
    if fault=='command':tape[3]['requested_command']=[.2,0.,0.]
    if fault=='comparison':saved[3]['comparison']['normalized_complete_decision_exact']=False
    if fault=='incomplete':tape[-1]['completed']=False
    if fault=='observed_state':saved[-1]['decision']['evidence']['observed_xy'][0]+=1
    if fault=='forecast':saved[-1]['decision']['new_selection']['prediction']=['changed']
    if fault=='expected':expected[-1]['candidate_selection_sha256']='wrong'
    if fault=='original_row':old[4]['pre_sample_index']+=1
    if fault=='receipt':saved[405]['decision']['new_selection']['sustained_hold_reorientation']['starting']=False
    with pytest.raises(ValueError):prefix.reconstruct(r,old,saved,tape,expected)


@pytest.mark.parametrize('fault',['incomplete','post','pre','earlier','boundary','short'])
def test_both_actual_command_prefixes_must_complete_and_match_until_declared_change(fault):
    r,_,_,old,_=fixture();new=deepcopy(old);new[-1]['requested_command']=r['candidate_requested_command']
    if fault=='incomplete':new[-1]['completed']=False
    if fault=='post':new[-1]['post_sample_index']-=1
    if fault=='pre':new[-1]['pre_sample_index']-=1
    if fault=='earlier':new[3]['requested_command']=[.2,0.,0.]
    if fault=='boundary':new[-1]['requested_command']=[0.,0.,-.45]
    if fault=='short':new.pop()
    with pytest.raises(ValueError):prefix.executed_boundary([old,new],r)


def physical_fixture(monkeypatch,tmp_path,fault=None):
    r,old,saved,tape,expected=fixture()
    prior=tmp_path/'prior';current=tmp_path/'current';root=tmp_path/'prefix';prior.mkdir();current.mkdir()
    actual=deepcopy(old)
    for i,row in enumerate(actual):row['decision']=deepcopy(saved[i]['decision'])
    new_tape=deepcopy(tape);new_tape[-1]['requested_command']=r['candidate_requested_command']
    raw=np.arange(21100,dtype=np.float64);other=raw.copy();other[21050:]+=100
    if fault=='physics':other[21049]+=1
    if fault=='intervention_trace':other=other[:21050]
    np.savez(prior/'physics_trace.npz',sample=raw);np.savez(current/'physics_trace.npz',sample=other)
    if fault=='decision':actual[-1]['decision']['evidence']['observed_xy'][0]+=1
    if fault=='completion':new_tape[-1]['completed']=False
    if fault=='public':saved[-1]['public_input_sha256']='different'
    if fault=='observation':actual[-1]['observation_index']-=1
    monkeypatch.setattr(prefix,'artifact_path',lambda parent,name:parent/name)
    monkeypatch.setattr(prefix,'read_rows',lambda p:iter(deepcopy(old if p==prior else actual if p==current else saved)))
    monkeypatch.setattr(prefix,'read_json',lambda p,n:deepcopy(tape if p==prior else new_tape))
    monkeypatch.setattr(prefix,'public_packets',lambda directory,frames:('packet_'+str(i) for i in range(frames)))
    monkeypatch.setattr(prefix.replay,'saved_inputs',lambda:dict(comparisons=expected))
    return prior,current,root,r


def test_physics_comparison_stops_at_intervention_but_requires_its_full_execution(monkeypatch,tmp_path):
    r=prefix.compare(*physical_fixture(monkeypatch,tmp_path))
    assert r['physical_prefix_samples']==21050 and r['common_prefix_frames']==407
    assert r['candidate_intervention_command_completed'] and not r['following_physical_outcomes_compared']


@pytest.mark.parametrize('fault',['physics','intervention_trace','decision','completion','public','observation'])
def test_bad_native_or_public_evidence_cannot_establish_intervention(fault,monkeypatch,tmp_path):
    with pytest.raises(ValueError):prefix.compare(*physical_fixture(monkeypatch,tmp_path,fault))


def test_public_packet_fingerprint_uses_raw_runner_order(monkeypatch):
    class Reader:
        def __init__(self,directory):pass
        def packet(self,i):return 'policy','primary','gyro',1500000000
    monkeypatch.setattr(prefix,'IntentReturnRGBDReplay',Reader)
    monkeypatch.setattr(prefix,'read_json',lambda *args:[{}])
    monkeypatch.setattr(prefix,'public_acquisition',lambda x:x)
    monkeypatch.setattr(prefix,'packet',lambda *args,**kwargs:('rgb','auxiliary'))
    assert list(prefix.public_packets('directory',1))==[prefix.fingerprint(('policy','primary','gyro','rgb','auxiliary',1500000000))]


def admission_fixture(monkeypatch,tmp_path,fault=None):
    report,old,saved,tape,expected=fixture()
    report.update(original_complete_decisions_reconstructed=True,candidate_matches_every_saved_selection=True,
        observed_map_and_contact_state_exact=True,prior_pending_model_forecasts_exact=True,
        model_state_sha256=prefix.replay.MODEL_SHA,model_state_unchanged=True,
        no_observation_after_changed_request_consumed=True,changed_command_executed=False,
        native_execution=False,unexecuted_outcomes_inferred=False,navigation_verified=False)
    result=dict(status='SUSTAINED_HOLD_REORIENTATION_RAW_PREFIX_V1_COMPLETE',native_execution=False,
        source_sha256={},artifact_sha256={},report=report)
    launch=dict(source_sha256={},saved_prefix_sha256=prefix.replay.SAVED_SHA,
        model_state_sha256=prefix.replay.MODEL_SHA,frames=407,
        input_admission=dict(native_result_sha256=prefix.replay.saved.prior.NATIVE_SHA))
    if fault=='status':result['status']='PARTIAL'
    if fault=='model':report['model_state_sha256']='0'*64
    if fault=='flag_type':report['model_state_unchanged']=1
    if fault=='future':report['no_observation_after_changed_request_consumed']=False
    if fault=='launch':launch['saved_prefix_sha256']='wrong'
    if fault=='public':saved[-1]['public_input_sha256']='wrong'
    calls=[]
    monkeypatch.setattr(prefix,'verify',lambda bindings:None)
    monkeypatch.setattr(prefix,'verify_artifacts',lambda root,bindings:calls.append((root,bindings)))
    def read(root,name):
        if name=='result.json':return deepcopy(result)
        if name=='launch.json':return deepcopy(launch)
        return deepcopy(tape)
    monkeypatch.setattr(prefix,'read_json',read)
    monkeypatch.setattr(prefix,'read_rows',lambda root:iter(deepcopy(saved if root==tmp_path else old)))
    monkeypatch.setattr(prefix.replay,'saved_inputs',lambda:dict(comparisons=expected))
    monkeypatch.setattr(prefix,'public_packets',lambda directory,frames:('packet_'+str(i) for i in range(frames)))
    return result,calls


def test_completion_admission_reconstructs_full_stream_and_binds_exact_launch(monkeypatch,tmp_path):
    result,calls=admission_fixture(monkeypatch,tmp_path)
    assert prefix.admit_prefix(tmp_path,result)==result['report']
    assert calls[0][1]['launch.json']==prefix.LAUNCH_SHA and len(calls)==2


@pytest.mark.parametrize('fault',['status','model','flag_type','future','launch','public'])
def test_completion_admission_rejects_incomplete_or_changed_scope(monkeypatch,tmp_path,fault):
    result,_=admission_fixture(monkeypatch,tmp_path,fault)
    with pytest.raises(ValueError):prefix.admit_prefix(tmp_path,result)
