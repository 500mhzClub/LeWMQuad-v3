"""Bounded replay consumes no physical future of the changed request."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
import pytest
from scripts import replay_go2_recent_qualified_direct_flow_maze03_prefix_v1 as runner
from lewm.tests.test_recent_qualified_direct_flow_prefix_development import row


@pytest.mark.parametrize('fault',[None,'public_mutation','model_changed'])
def test_replay_stops_before_fourth_observation_and_retains_comparator_failure(monkeypatch,tmp_path,fault):
    source=tmp_path/'input';output=tmp_path/'output';output.mkdir();(output/runner.DECISIONS).write_bytes(b'')
    monkeypatch.setattr(runner,'INPUT',source);monkeypatch.setattr(runner,'OUTPUT',output)
    models=[];packets=[];records=[];stored=[];current=[];checks=[]
    for i in range(3):
        old,new=row(i,previous=None if i==0 else i-1,attempt=i==2)
        old.update(tick=i,failure=None);new.update(tick=i,failure=None,evidence={'fixture':i})
        if i==2:new['requested_command']=[.16,0.,.45]
        # The new typed evidence is also present in the original prefix fixture.
        old['evidence']={'fixture':i}
        records.append(dict(tick=i,observation_index=i,pre_sample_index=749+50*i,decision=old));current.append(new)
    def read_rows(path):
        assert path==source/runner.CASE[0]
        yield from records
        pytest.fail('consumed observation after changed command')
    monkeypatch.setattr(runner,'read_rows',read_rows)
    def packet(i):
        packets.append(i)
        return {'frame':i},{},{},1_500_000_000+100_000_000*i
    monkeypatch.setattr(runner,'IntentReturnRGBDReplay',lambda p:SimpleNamespace(frames=[None]*1217,packet=packet))
    monkeypatch.setattr(runner,'read_json',lambda p,n:[{}]*1217 if n=='auxiliary_camera_audit.json' else
        [dict(completed=True,requested_command=[0.,0.,0.])]*1216)
    monkeypatch.setattr(runner,'packet',lambda *a,**k:({},{}));monkeypatch.setattr(runner,'public_acquisition',lambda r:r)
    model=SimpleNamespace(state_dict=lambda:{},parameters=lambda:[])
    monkeypatch.setattr(runner,'load_assigned',lambda *a:(model,runner.CASE[3],runner.CASE[2]))
    calls=[]
    def state(value):
        calls.append(True)
        return '0'*64 if fault=='model_changed' and len(calls)>1 else runner.MODEL_STATE
    monkeypatch.setattr(runner,'state_digest',state)
    monkeypatch.setattr(runner,'ArticulatedCollisionGeometry',lambda *a:object())
    def controller(*a,**k):
        models.append(a[0]);assert a[0] is model
        def observe(policy,*args,**kwargs):
            i=policy['frame']
            if fault=='public_mutation':policy['changed']=True
            return deepcopy(current[i])
        return SimpleNamespace(observe=observe)
    monkeypatch.setattr(runner,'RecentQualifiedDirectFlowController',controller)
    monkeypatch.setattr(runner,'current_dual_camera_pose',lambda *a,**k:checks.append('raw'))
    monkeypatch.setattr(runner,'current_measured_floor_pose',lambda *a,**k:checks.append('registered'))
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda *a:SimpleNamespace(free=100*1024**3))
    @contextmanager
    def writer(path):yield stored.append
    monkeypatch.setattr(runner,'writer',writer)
    if fault is None:
        report=runner.replay({'correction_admission':{}})
        assert report['frames']==3 and report['first_requested_command_difference']==2
        assert report['raw_model_forecast_comparisons']==3 and report['model_state_unchanged']
        assert packets==[0,1,2] and len(stored)==3 and checks==['raw','registered']*3
        assert not report['following_recorded_observations_consumed'] and not report['native_execution']
    else:
        with pytest.raises(ValueError):runner.replay({'correction_admission':{}})
        if fault=='public_mutation':
            assert packets==[0] and 'comparison_failure' in stored[0]
        else:assert packets==[0,1,2] and len(stored)==3


@pytest.mark.parametrize('fault',[None,'native_identity','artifact_identity','benchmark','context_error','context_mutation'])
def test_verification_keeps_fixed_completed_native_identity_and_original_context(monkeypatch,fault):
    launch=dict(native_result_sha256='a'*64,replay_input_bindings={**runner.INPUT_BINDINGS,'result.json':'a'*64},
        verification_benchmark_result_sha256=runner.BENCHMARK_SHA)
    calls=[]
    if fault=='native_identity':launch['native_result_sha256']='0'*64
    elif fault=='artifact_identity':launch['replay_input_bindings']['result.json']='0'*64
    elif fault=='benchmark':launch['verification_benchmark_result_sha256']='0'*64
    monkeypatch.setattr(runner,'admit_benchmark',lambda:calls.append('benchmark'))
    def scope(function,digest,value):
        assert function is runner.verify_input_context and digest is runner.digest and value is launch
        calls.append('original_context')
        if fault=='context_error':raise ValueError('original verification failed')
        if fault=='context_mutation':value['changed']=True
        return None,{}
    monkeypatch.setattr(runner,'verify_with_scoped_digests',scope)
    if fault is None:
        runner.verify_inputs(launch);assert calls==['benchmark','original_context']
    else:
        with pytest.raises(ValueError):runner.verify_inputs(launch)
        if fault in ('native_identity','artifact_identity','benchmark'):assert not calls


def admitted_fixture():
    from lewm.independent_reactive_floor_transport_study_development import OUTCOME_KEYS
    outcomes={k:False for k in OUTCOME_KEYS}
    outcomes['native_evaluation']={}
    outcomes['renderer_capture_audit']={}
    outcomes['hard_measurement_failed_frames']=[]
    audit=dict(layout_index=3,raw_sensor_reconstruction_pass=True,raw_command_audit_pass=True,
        raw_model_command_replay_pass=True,model_state_unchanged=True,**outcomes)
    prefix=dict(common_prefix_frames=265,first_intervention_frame=264,physical_prefix_samples=13950,
        physical_and_public_prefix_exact=True,all_preintervention_observed_state_exact=True,
        all_preintervention_requested_commands_exact=True,
        complete_candidate_decisions_match_prospective_prefix=True,raw_model_forecast_comparisons=261,
        all_compared_raw_model_forecasts_exact=True,candidate_intervention_command_completed=True,
        original_intervention_command=[0.,0.,0.],candidate_intervention_command=[0.,0.,.45],
        following_physical_outcomes_compared=False,unexecuted_outcomes_inferred=False)
    record=dict(case=runner.CASE[0],layout_index=3,
        status='DIRECT_FLOW_MAZE03_COLLECTED_AND_RAW_AUDITED',model_state_unchanged=True,
        collection=dict(decisions=1217,completed_ticks=1216,physics_samples=61550),
        prefix_comparison=prefix,**outcomes)
    launch=dict(planned_case=list(runner.CASE),implementation_class='DirectFlowFloorTransportController',
        scene_specification=runner.specification(3),public_mission=runner.public_mission(3),
        prefix_report=dict(model_state_sha256=runner.MODEL_STATE),source_sha256={})
    result=dict(status='DIRECT_FLOW_MAZE03_PILOT_V1_COMPLETE',conditions=[record],
        source_sha256={},artifact_sha256=dict(runner.INPUT_BINDINGS),model_training=False,
        learned_cohort_result_sha256='a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720')
    return result,launch,audit


@pytest.mark.parametrize('fault',[None,'incomplete','height','sources','collection','raw_audit',
    'outcome','uncompleted_intervention','prefix_count',*runner.INPUT_BINDINGS])
def test_completed_native_admission_rejects_provisional_or_changed_inputs(fault):
    result,launch,audit=admitted_fixture()
    if fault=='incomplete':result['status']='DIRECT_FLOW_MAZE03_TERMINAL_AUDIT_REQUIRED'
    elif fault=='height':launch['implementation_class']='PartialHeightDirectFlowController'
    elif fault=='sources':result['source_sha256']={'changed':'0'*64}
    elif fault=='collection':result['conditions'][0]['collection']['decisions']-=1
    elif fault=='raw_audit':audit['raw_sensor_reconstruction_pass']=False
    elif fault=='outcome':audit['strict_physical_visibility_pass']=True
    elif fault=='uncompleted_intervention':result['conditions'][0]['prefix_comparison']['candidate_intervention_command_completed']=False
    elif fault=='prefix_count':result['conditions'][0]['prefix_comparison']['physical_prefix_samples']-=1
    elif fault in runner.INPUT_BINDINGS:result['artifact_sha256'][fault]='0'*64
    if fault is None:runner.admit_native(result,launch,audit)
    else:
        with pytest.raises(ValueError):runner.admit_native(result,launch,audit)
