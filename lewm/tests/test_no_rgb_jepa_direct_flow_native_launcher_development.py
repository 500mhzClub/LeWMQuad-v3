"""Fixed treatment, closed queue, fresh audit model, and retained failed evidence."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from scripts import run_go2_no_rgb_jepa_direct_flow_maze02_pilot_v1 as native
from scripts import no_rgb_jepa_direct_flow_native_inputs_development as inputs
from lewm.tests.test_no_rgb_jepa_direct_flow_native_prefix_development import fixture as prefix_fixture


def fixture():
    prefix=prefix_fixture()[0]
    audit=dict(layout_index=2,direct_corner_flow_missingness_fallback_enabled=True,
        raw_sensor_reconstruction_pass=True,raw_command_audit_pass=True,raw_model_command_replay_pass=True,
        model_state_unchanged=True,native_evaluation=dict(native_round_trip_candidate_pass=False),
        strict_physical_visibility_pass=True,hard_measurement_failed_frames=[],renderer_capture_audit={},verified_round_trip=False)
    receipt=dict(common_prefix_frames=860,first_intervention_frame=859,physical_prefix_samples=43700,
        original_forecasts_compared=856,physical_and_public_prefix_exact=True,
        all_preintervention_requested_commands_exact=True,complete_candidate_decisions_match_prospective_prefix=True,
        candidate_intervention_command_completed=True,candidate_intervention_command=[0.,0.,0.],
        intervention_command_changed=False,observer_and_full_controller_intervention=True,navigation_verified=False,
        following_physical_outcomes_compared=False,unexecuted_outcomes_inferred=False)
    record=dict(status=native.WORKER_STATUS,case=native.CASE[0],layout_index=2,
        model_name=native.CASE[4],condition='jepa',variant='no_rgb',model_state_sha256=inputs.replay.MODEL_SHA,
        model_state_unchanged=True,collection=dict(status='NO_RGB_JEPA_DIRECT_FLOW_MAZE02_TERMINAL_AUDIT_REQUIRED',
        direct_corner_flow_missingness_fallback_enabled=True),prefix_comparison=receipt,
        **{k:deepcopy(audit[k]) for k in native.OUTCOME_KEYS})
    return record,audit,prefix


def test_complete_scientific_failure_is_retained_without_claiming_movement():
    native.require_worker(*fixture())


@pytest.mark.parametrize('fault',['model','variant','condition','collection','raw','intervention','frames','physics',
    'forecasts','command','counterfactual','false_success','hold_is_movement','prefix_navigation'])
def test_incomplete_changed_or_overclaimed_native_result_is_rejected(fault):
    record,audit,prefix=fixture();p=record['prefix_comparison']
    if fault=='model':record['model_state_sha256']='wrong'
    elif fault=='variant':record['variant']='full'
    elif fault=='condition':record['condition']='supervised_rollout'
    elif fault=='collection':record['collection']['direct_corner_flow_missingness_fallback_enabled']=False
    elif fault=='raw':audit['raw_model_command_replay_pass']=False
    elif fault=='intervention':p['candidate_intervention_command_completed']=False
    elif fault=='frames':p['common_prefix_frames']=859
    elif fault=='physics':p['physical_prefix_samples']=43699
    elif fault=='forecasts':p['original_forecasts_compared']=855
    elif fault=='command':p['candidate_intervention_command']=[.2,0.,0.]
    elif fault=='counterfactual':p['following_physical_outcomes_compared']=True
    elif fault=='false_success':record['verified_round_trip']=audit['verified_round_trip']=True
    elif fault=='hold_is_movement':p['intervention_command_changed']=True
    else:p['navigation_verified']=True
    with pytest.raises(ValueError):native.require_worker(record,audit,prefix)


def test_incompatible_model_wrapper_rejected_before_scene_creation(monkeypatch):
    monkeypatch.setattr(native,'load_assigned',lambda *a:(SimpleNamespace(training=False),'jepa','no_rgb'))
    with pytest.raises(ValueError,match='assigned'):native.assigned_model(dict(input_admission=dict(correction_admission={})))


@pytest.mark.parametrize('replay_live,queue_live',[(True,False),(False,True)])
def test_live_owners_prevent_input_admission_before_any_artifact_access(monkeypatch,replay_live,queue_live):
    monkeypatch.setattr(inputs,'owner_live',lambda owner:replay_live)
    def queue_ended():
        if queue_live:raise ValueError('queue live')
    monkeypatch.setattr(inputs.queue,'owners_ended',queue_ended)
    monkeypatch.setattr(inputs.prefix,'admit_prefix',lambda *a:pytest.fail('must reject before prefix admission'))
    with pytest.raises(ValueError):inputs.admit('p',{},'b',{})


def test_source_preflight_does_not_admit_inputs_load_models_or_create_output(monkeypatch,tmp_path):
    monkeypatch.setattr(native,'OUTPUT',tmp_path/'uncreated')
    monkeypatch.setattr(native,'validate_root',lambda *a,**k:None)
    monkeypatch.setattr(inputs,'prepared_sources',lambda *a:{})
    monkeypatch.setattr(native,'hardware',lambda:dict(memory_available_bytes=64*1024**3,artifact_free_bytes=100*1024**3))
    monkeypatch.setattr(inputs,'admit',lambda *a:pytest.fail('no input admission'))
    monkeypatch.setattr(native,'assigned_model',lambda *a:pytest.fail('no model'))
    monkeypatch.setattr(native,'create_output',lambda *a:pytest.fail('no runtime output'))
    monkeypatch.setattr('sys.argv',['native','--source-preflight-only'])
    native.main();assert not native.OUTPUT.exists()


def admission_fixture(monkeypatch):
    prefix=fixture()[2];events=[]
    prefix_ids={'result.json':'p','launch.json':'l','context_decisions.jsonl.gz':'d','report.json':'r'}
    observer_ids={'result.json':inputs.replay.OBSERVER_SHA};old_ids={'original':'o'}
    original=dict(input_admission=dict(correction_admission={'original_model':'same'}))
    prefix_result=dict(report=prefix,artifact_sha256={k:v for k,v in prefix_ids.items() if k!='result.json'})
    prefix_launch=dict(observer_artifact_sha256=observer_ids,input_artifact_sha256=old_ids)
    completed=dict(ordered_waiter_result_sha256={'frontier':'f','hold':'h','contact':'c'},adapter_batch_result_sha256='b',
        complete_original_native_queue_authenticated=True,original_queue_owners_ended=True,
        original_native_input_admissions_fully_reexecuted=False)
    monkeypatch.setattr(inputs,'owners_ended',lambda:events.append('owners'))
    monkeypatch.setattr(inputs,'verify',lambda *a:None)
    monkeypatch.setattr(inputs,'verify_artifacts',lambda *a:None)
    monkeypatch.setattr(inputs.prefix,'admit_prefix',lambda *a:deepcopy(prefix))
    monkeypatch.setattr(inputs.replay,'verify_inputs',lambda *a,**k:events.append(('full_model_inputs',k)))
    monkeypatch.setattr(inputs.replay.batch,'verify_inputs',lambda *a,**k:events.append(('bound_model_inputs',k)))
    def queue_admit(ids,**kwargs):
        assert ids==completed['ordered_waiter_result_sha256'] and kwargs['adapter_batch_result_sha256']=='b'
        events.append(('queue',kwargs['full']));return deepcopy(completed)
    monkeypatch.setattr(inputs.queue,'admit',queue_admit)
    def read(root,name):
        if root==inputs.replay.OUTPUT:return deepcopy(prefix_result if name=='result.json' else prefix_launch)
        if root==inputs.replay.batch.OUTPUT:return deepcopy(original)
        raise AssertionError((root,name))
    monkeypatch.setattr(inputs,'read_json',read)
    return completed,events


def test_own_model_inputs_fully_rechecked_and_queue_used_only_for_completion(monkeypatch):
    completed,events=admission_fixture(monkeypatch)
    result=inputs.admit('p',completed['ordered_waiter_result_sha256'],'b',{})
    assert ('full_model_inputs',{'full':True}) in events
    assert all(full is False for name,full in [e for e in events if isinstance(e,tuple) and e[0]=='queue'])
    assert result['queue_scope']=='operational completion only; no queued policy adopted'
    assert result['final_independent_population_policy_review_performed'] is False


@pytest.mark.parametrize('fault',['correction','prefix','artifacts','observer','episode','queue_scope','policy_claim'])
def test_bound_input_substitutions_are_rejected(monkeypatch,fault):
    completed,_=admission_fixture(monkeypatch)
    result=inputs.admit('p',completed['ordered_waiter_result_sha256'],'b',{})
    if fault=='correction':result['correction_admission']={}
    elif fault=='prefix':result['prefix_report']['boundary_requested_command']=[.2,0.,0.]
    elif fault=='artifacts':result['prefix_artifact_sha256'].pop('report.json')
    elif fault=='observer':result['observer_artifact_sha256']={}
    elif fault=='episode':result['original_episode_artifact_sha256']={}
    elif fault=='queue_scope':result['queue_scope']='final policy selected'
    else:result['final_independent_population_policy_review_performed']=True
    with pytest.raises(ValueError):inputs.verify_bound(result,{})


@pytest.mark.parametrize('failed_stage',[None,'collection','audit','prefix','verification'])
def test_worker_uses_fresh_audit_model_and_retains_failed_artifacts(monkeypatch,tmp_path,failed_stage):
    expected,report,prefix=fixture();models=[];calls=[]
    launch=dict(source_sha256={native.PROTOCOL:'a'*64},input_admission={'prefix_report':prefix})
    monkeypatch.setattr(native,'OUTPUT',tmp_path)
    monkeypatch.setattr(native,'read_json',lambda *a:launch)
    monkeypatch.setattr(native,'verify_artifacts',lambda *a:None)
    def verify(*a,**kwargs):
        calls.append('verify')
        if failed_stage=='verification' and calls.count('verify')==2:raise ValueError('synthetic verification failure')
    monkeypatch.setattr(native,'verify_inputs',verify)
    def model(*a):
        obj=SimpleNamespace(state_dict=lambda:{});models.append(obj);return obj
    monkeypatch.setattr(native,'assigned_model',model)
    monkeypatch.setattr(native,'state_digest',lambda *a:inputs.replay.MODEL_SHA)
    monkeypatch.setattr(native,'ArticulatedCollisionGeometry',lambda *a:object())
    def collect(index,definition,**kwargs):
        assert kwargs['model'] is models[0] and index==2 and kwargs['condition']=='jepa' and kwargs['variant']=='no_rgb'
        path=tmp_path/native.CASE[0];path.mkdir();(path/'retained_raw.json').write_text('{}')
        np.savez(path/'physics_trace.npz',physics_contact=np.zeros(2))
        if failed_stage=='collection':raise ValueError('synthetic collection failure')
        return deepcopy(expected['collection'])
    def audit(index,result,definition,**kwargs):
        assert kwargs['model'] is models[1] and models[1] is not models[0]
        if failed_stage=='audit':raise ValueError('synthetic audit failure')
        return deepcopy(report)
    def compare(*a):
        if failed_stage=='prefix':raise ValueError('synthetic prefix failure')
        return deepcopy(expected['prefix_comparison'])
    monkeypatch.setattr(native,'collect',collect);monkeypatch.setattr(native,'audit',audit);monkeypatch.setattr(native,'compare',compare)
    monkeypatch.setattr(native,'artifacts',lambda *a:['retained_raw.json','physics_trace.npz'])
    monkeypatch.setattr(native,'case_readout',lambda *a:{'verified_round_trip':False})
    result=native.worker('b'*64)
    assert (tmp_path/native.CASE[0]/'retained_raw.json').is_file()
    assert (tmp_path/(native.CASE[0]+'_worker_terminal.json')).is_file()
    assert len(models)==(1 if failed_stage=='collection' else 2)
    if failed_stage is None:
        assert result['status']==native.WORKER_STATUS and result['verified_round_trip'] is False
    else:
        assert result['status']=='NO_RGB_JEPA_DIRECT_FLOW_MAZE02_WORKER_FAILED'
        assert 'synthetic '+failed_stage+' failure' in result['failure']
    if failed_stage!='collection':assert native.CASE[0]+'/retained_raw.json' in result['artifact_sha256']
    if failed_stage not in ('collection','audit'):assert native.CASE[0]+'_audit.json' in result['artifact_sha256']
