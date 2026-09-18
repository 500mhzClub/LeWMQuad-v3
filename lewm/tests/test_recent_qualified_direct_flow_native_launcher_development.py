"""Unchanged physical/evaluation code and preserved evidence across worker failures."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from lewm.independent_floor_transport_study_development import MODEL_STATE
from scripts import run_go2_recent_qualified_direct_flow_maze03_pilot_v1 as runner


@pytest.mark.parametrize('failed_stage',[None,'audit','prefix','verification','partial_boundary'])
def test_worker_uses_fresh_models_and_retains_evidence_after_failure(monkeypatch,tmp_path,failed_stage):
    monkeypatch.setattr(runner,'OUTPUT',tmp_path)
    launch=dict(robot_urdf_sha256='a'*64,source_sha256={runner.PROTOCOL:'b'*64},correction_admission={},
        prefix_report={'model_state_sha256':MODEL_STATE}, predecessor_admission={'fixture':True})
    monkeypatch.setattr(runner,'read_json',lambda *a:launch)
    monkeypatch.setattr(runner,'verify_artifacts',lambda *a:None)
    calls=[]; models=[]
    def verify(*a):
        calls.append('verify')
        if failed_stage=='verification' and calls.count('verify')==2: raise ValueError('synthetic verification failure')
    monkeypatch.setattr(runner,'verify_inputs',verify)
    digest=runner.digest; monkeypatch.setattr(runner,'digest',lambda p:'a'*64 if p==runner.URDF else digest(p))
    monkeypatch.setattr(runner,'state_digest',lambda d:MODEL_STATE)
    monkeypatch.setattr(runner,'ArticulatedCollisionGeometry',lambda *a:object())
    def load(*a):
        models.append(SimpleNamespace(state_dict=lambda:{})); return models[-1],runner.CASE[3],runner.CASE[2]
    monkeypatch.setattr(runner,'load_assigned',load)
    def collect(index,definition,**kwargs):
        assert index==3 and kwargs['model'] is models[0] and kwargs['episode_name']==runner.CASE[0]
        path=tmp_path/runner.CASE[0]; path.mkdir(); (path/'retained_raw.json').write_text('{}'); calls.append('collect')
        return {'fixture':True}
    def audit(index,result,definition,**kwargs):
        assert kwargs['model'] is models[1] and models[1] is not models[0]; calls.append('audit')
        if failed_stage=='audit': raise ValueError('synthetic audit failure')
        return dict(verified_round_trip=False,native_evaluation={},strict_physical_visibility_pass=False,
            hard_measurement_failed_frames=[],renderer_capture_audit={})
    def compare(*a):
        calls.append('prefix')
        if failed_stage=='prefix': raise ValueError('synthetic prefix failure')
        return {'physical_and_public_prefix_exact':True,
            'candidate_intervention_command_completed':failed_stage!='partial_boundary'}
    monkeypatch.setattr(runner,'collect',collect); monkeypatch.setattr(runner,'audit',audit)
    monkeypatch.setattr(runner,'compare',compare); monkeypatch.setattr(runner,'artifacts',lambda *a:['retained_raw.json'])
    record=runner.worker('c'*64)
    assert len(models)==2 and runner.CASE[0]+'/retained_raw.json' in record['artifact_sha256']
    assert (tmp_path/(runner.CASE[0]+'_worker_terminal.json')).is_file()
    if failed_stage is None:
        assert record['status']=='RECENT_QUALIFIED_DIRECT_FLOW_MAZE03_COLLECTED_AND_RAW_AUDITED'
        assert not record['verified_round_trip'] and record['model_state_unchanged']
        assert record['recent_qualified_anchor_enabled']
    else:
        assert record['status']=='RECENT_QUALIFIED_DIRECT_FLOW_MAZE03_WORKER_FAILED'
        if failed_stage=='partial_boundary':assert 'prospective intervention must have completed' in record['failure']
        else:assert 'synthetic '+failed_stage+' failure' in record['failure']
    if failed_stage!='audit': assert runner.CASE[0]+'_audit.json' in record['artifact_sha256']
    if failed_stage in (None,'verification','partial_boundary'):
        assert runner.CASE[0]+'_prefix_comparison.json' in record['artifact_sha256']


def test_worker_requires_completed_predecessor_before_any_model_or_collection(monkeypatch,tmp_path):
    monkeypatch.setattr(runner,'OUTPUT',tmp_path)
    monkeypatch.setattr(runner,'verify_artifacts',lambda *a:None)
    monkeypatch.setattr(runner,'read_json',lambda *a:{})
    monkeypatch.setattr(runner,'load_assigned',lambda *a:pytest.fail('model loaded without predecessor'))
    monkeypatch.setattr(runner,'collect',lambda *a,**k:pytest.fail('collection without predecessor'))
    record=runner.worker('a'*64)
    assert record['status']=='RECENT_QUALIFIED_DIRECT_FLOW_MAZE03_WORKER_FAILED'
    assert 'completed predecessor admission required' in record['failure']
    assert record['artifact_sha256']=={}


def context():
    report=dict(frames=1207,maximum_frames=1207,first_requested_command_difference=1206,
        exact_original_decisions=1193,raw_model_forecast_comparisons=1203,
        extra_reference_attempts=1,extra_qualified_references=1,first_reference_attempt=1193,
        first_qualified_reference=1193,first_decision_difference=1193,
        final_requested_command=[0.,0.,.45],prior_requested_command=[0.,0.,0.],
        final_terminal=None,final_failure=None,prior_terminal='SENSOR_OR_MODEL_FAILURE')
    prefix = dict(verification_benchmark_result_sha256=runner.BENCHMARK_SHA,
        native_result_sha256=runner.PRIOR_SHA,
        replay_input_bindings={'result.json':runner.PRIOR_SHA},correction_admission={'model':'fixture'})
    launch = dict(verification_benchmark_result_sha256=runner.BENCHMARK_SHA,
        prior_native_result_sha256=runner.PRIOR_SHA,prospective_prefix_result_sha256=runner.PREFIX_SHA,
        prefix_artifact_sha256={'result.json':runner.PREFIX_SHA},
        prior_artifact_sha256=deepcopy(prefix['replay_input_bindings']),
        correction_admission=deepcopy(prefix['correction_admission']),prefix_report=report,
        planned_case=list(runner.CASE),scene_specification=runner.specification(3),
        public_mission=runner.public_mission(3),implementation_class='RecentQualifiedDirectFlowController',
        recent_qualified_anchor_enabled=True)
    return launch,prefix


@pytest.mark.parametrize('fault',[None,'benchmark','native','prefix','prefix_binding','native_binding',
    'original_error','input_mutation','wrong_return'])
def test_fresh_scope_rejects_wrong_identities_and_verifier_mutation(monkeypatch,fault):
    launch,_ = context(); calls=[]
    if fault in ('benchmark','native','prefix'):
        launch[{'benchmark':'verification_benchmark_result_sha256','native':'prior_native_result_sha256',
            'prefix':'prospective_prefix_result_sha256'}[fault]]='0'*64
    elif fault=='prefix_binding': launch['prefix_artifact_sha256']['result.json']='0'*64
    elif fault=='native_binding': launch['prior_artifact_sha256']['result.json']='0'*64
    before=deepcopy(launch)
    monkeypatch.setattr(runner,'admit_benchmark',lambda:calls.append('benchmark'))
    def scope(function,digest,value):
        assert function is runner.verify_input_context and digest is runner.digest and value is launch
        calls.append('scope')
        if fault=='original_error': raise ValueError('original context rejected')
        if fault=='input_mutation': value['changed']=True
        return ('bad' if fault=='wrong_return' else None),{}
    monkeypatch.setattr(runner,'verify_with_scoped_digests',scope)
    if fault is None:
        runner.verify_inputs(launch)
        assert calls==['benchmark','scope'] and launch==before
    else:
        with pytest.raises(ValueError): runner.verify_inputs(launch)
        if fault in ('benchmark','native','prefix','prefix_binding','native_binding'): assert not calls


@pytest.mark.parametrize('fault',[None,'benchmark','native','bindings','model',
    'report','fixed_definition','case','scene','mission','implementation','anchor_disabled','hold_disabled','original_error'])
def test_context_executes_all_original_prefix_identity_conditions(monkeypatch,fault):
    launch,prefix=context(); report=deepcopy(launch['prefix_report']); calls=[]
    if fault in ('benchmark','native'):
        prefix[{'benchmark':'verification_benchmark_result_sha256','native':'native_result_sha256'}[fault]]='0'*64
    elif fault=='bindings': prefix['replay_input_bindings']['unexpected']='0'*64
    elif fault=='model': launch['correction_admission']['model']='different'
    elif fault=='report': launch['prefix_report']['frames']=180
    elif fault=='fixed_definition': launch['prefix_report']['frames']=report['frames']=1208
    elif fault=='case': launch['planned_case'][1]=2
    elif fault=='scene': launch['scene_specification']=runner.specification(2)
    elif fault=='mission': launch['public_mission']={}
    elif fault=='implementation': launch['implementation_class']='other'
    elif fault=='anchor_disabled': launch['recent_qualified_anchor_enabled']=False
    elif fault=='hold_disabled': launch['partial_floor_height_constraint_enabled']=True
    monkeypatch.setattr(runner,'verify',lambda value:calls.append('environment'))
    monkeypatch.setattr(runner,'verify_artifacts',lambda root,ids:calls.append(('artifacts',root)))
    monkeypatch.setattr(runner,'read_json',lambda root,name:prefix if name=='launch.json' else {'report':report})
    def original(value):
        assert value is prefix; calls.append('original')
        if fault=='original_error': raise ValueError('original failed')
    monkeypatch.setattr(runner,'verify_prefix_context',original)
    if fault is None:
        runner.verify_input_context(launch)
        assert calls==['environment',('artifacts',runner.PREFIX),('artifacts',runner.PRIOR),'original']
    else:
        with pytest.raises(ValueError): runner.verify_input_context(launch)


def test_fixed_completed_inputs_and_fresh_maze3_case():
    assert runner.CASE==('full_jepa_recent_qualified_direct_flow_maze_03',3,'full','jepa','seed_2026091001_full_jepa')
    assert runner.PRIOR_CASE[0]=='full_jepa_direct_flow_maze_03'
    assert runner.PREFIX.name=='go2_recent_qualified_direct_flow_maze03_prefix_v1_attempt_001'
    assert runner.PREFIX_SHA=='16c6917d2e4c2b728bd08a330290e141a93a500f9f28b7a3abd27e9d4f51926a'
    assert runner.PRIOR_SHA=='6be6aa6e60be4b1a9e3d9b79aa3b55e4becc3265fe208ce900c8e1962c3db3ea'
    assert runner.OUTPUT not in (runner.PRIOR,runner.PREFIX)
