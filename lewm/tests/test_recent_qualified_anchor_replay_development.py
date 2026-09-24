"""Bounded replay consumes no physical future of the changed request."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
import pytest
from scripts import replay_go2_recent_qualified_anchor_prefix_v1 as runner
from lewm.tests.test_recent_qualified_anchor_prefix_development import row


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
    monkeypatch.setattr(runner,'IntentReturnRGBDReplay',lambda p:SimpleNamespace(frames=[None]*656,packet=packet))
    monkeypatch.setattr(runner,'read_json',lambda p,n:[{}]*656 if n=='auxiliary_camera_audit.json' else
        [dict(completed=True,requested_command=[0.,0.,0.])]*655)
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
    monkeypatch.setattr(runner,'RecentQualifiedAnchorController',controller)
    monkeypatch.setattr(runner,'current_dual_camera_pose',lambda *a,**k:checks.append('raw'))
    monkeypatch.setattr(runner,'current_partial_height_pose',lambda *a,**k:checks.append('registered'))
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
    launch=dict(native_result_sha256=runner.INPUT_SHA,replay_input_bindings={'result.json':runner.INPUT_SHA},
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
