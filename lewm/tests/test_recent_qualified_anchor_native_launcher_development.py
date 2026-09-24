"""Unchanged physical/evaluation code and preserved evidence across worker failures."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from lewm.independent_floor_transport_study_development import MODEL_STATE
from scripts import run_go2_recent_qualified_anchor_maze01_pilot_v1 as runner


@pytest.mark.parametrize('failed_stage',[None,'audit','prefix','verification'])
def test_worker_uses_fresh_models_and_retains_evidence_after_failure(monkeypatch,tmp_path,failed_stage):
    monkeypatch.setattr(runner,'OUTPUT',tmp_path)
    launch=dict(robot_urdf_sha256='a'*64,source_sha256={runner.PROTOCOL:'b'*64},correction_admission={},
        prefix_report={'model_state_sha256':MODEL_STATE})
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
        assert index==1 and kwargs['model'] is models[0] and kwargs['episode_name']==runner.CASE[0]
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
        return {'physical_and_public_prefix_exact':True}
    monkeypatch.setattr(runner,'collect',collect); monkeypatch.setattr(runner,'audit',audit)
    monkeypatch.setattr(runner,'compare',compare); monkeypatch.setattr(runner,'artifacts',lambda *a:['retained_raw.json'])
    record=runner.worker('c'*64)
    assert len(models)==2 and runner.CASE[0]+'/retained_raw.json' in record['artifact_sha256']
    assert (tmp_path/(runner.CASE[0]+'_worker_terminal.json')).is_file()
    if failed_stage is None:
        assert record['status']=='RECENT_QUALIFIED_ANCHOR_MAZE01_COLLECTED_AND_RAW_AUDITED'
        assert not record['verified_round_trip'] and record['model_state_unchanged']
        assert record['recent_qualified_anchor_enabled']
    else:
        assert record['status']=='RECENT_QUALIFIED_ANCHOR_MAZE01_WORKER_FAILED'
        assert 'synthetic '+failed_stage+' failure' in record['failure']
    if failed_stage!='audit': assert runner.CASE[0]+'_audit.json' in record['artifact_sha256']
    if failed_stage in (None,'verification'):
        assert runner.CASE[0]+'_prefix_comparison.json' in record['artifact_sha256']


def context():
    report=dict(frames=646,maximum_frames=646,first_requested_command_difference=645,
        exact_original_decisions=635,raw_model_forecast_comparisons=642,
        extra_reference_attempts=2,extra_qualified_references=1,first_reference_attempt=635,
        first_qualified_reference=635,first_decision_difference=635,
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
        planned_case=list(runner.CASE),scene_specification=runner.specification(1),
        public_mission=runner.public_mission(1),implementation_class='RecentQualifiedAnchorController',
        recent_qualified_anchor_enabled=True,partial_floor_height_constraint_enabled=True)
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
    elif fault=='fixed_definition': launch['prefix_report']['frames']=report['frames']=647
    elif fault=='case': launch['planned_case'][1]=2
    elif fault=='scene': launch['scene_specification']=runner.specification(2)
    elif fault=='mission': launch['public_mission']={}
    elif fault=='implementation': launch['implementation_class']='other'
    elif fault=='anchor_disabled': launch['recent_qualified_anchor_enabled']=False
    elif fault=='hold_disabled': launch['partial_floor_height_constraint_enabled']=False
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


def test_fixed_completed_inputs_and_fresh_maze1_case():
    assert runner.CASE==('full_jepa_recent_qualified_anchor_maze_01',1,'full','jepa','seed_2026091001_full_jepa')
    assert runner.PRIOR_CASE[0]=='full_jepa_partial_floor_height_maze_01'
    assert runner.PREFIX.name=='go2_recent_qualified_anchor_prefix_v1_attempt_001'
    assert runner.PREFIX_SHA=='6e25c6c561473b60966a1a47388a01b48ab3e547da0c8ab13aa1e267b9fad302'
    assert runner.PRIOR_SHA=='32edbb748e04e18816e0b0fb265f465706ac8c8684849d73a091321f79a3b07f'
    assert runner.OUTPUT not in (runner.PRIOR,runner.PREFIX)
