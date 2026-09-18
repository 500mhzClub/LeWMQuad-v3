"""Exact loop reuse, full synthetic paired histories and predecessor rejection."""
from copy import deepcopy
from itertools import islice
import json
import sys
from types import SimpleNamespace
import pytest

from scripts import replay_go2_scoped_batched_footprint_late_history_v1 as replay
from lewm import scoped_batched_footprint_controller_development as combined
from lewm.tests.test_scoped_footprint_late_history_replay_development import synthetic


def combined_fixture(monkeypatch,tmp_path,fault=None):
    previous=replay.previous
    references,tape,calls,models=synthetic(monkeypatch,tmp_path,fault)
    fake_scoped=previous.ScopedFootprintAnchoredController
    class Baseline(fake_scoped):
        index=0
    class Candidate(Baseline):
        index=1
        def observe(self,*args,**kwargs):
            return super().observe(*args,**kwargs) | dict(controller=combined.CONTROLLER,
                **{combined.FLAG:True,'batched_retained_floor_queries_enabled':True})
    monkeypatch.setattr(replay,'ScopedFootprintAnchoredController',Baseline)
    monkeypatch.setattr(replay,'ScopedBatchedFootprintController',Candidate)
    monkeypatch.setattr(replay,'OUTPUT',tmp_path)
    # The fake memory has no robot patch stores. Actual two-tag normalization
    # and geometry equivalence are exercised by the composition tests.
    monkeypatch.setattr(replay,'normalized_state_tree',previous.state_tree)
    for i,row in enumerate(islice(previous.profile.read_rows(None),previous.FRAMES)):
        decision=row['decision'] | {'controller':previous.CONTROLLER,previous.FLAG:True}
        references[i]['candidate_decision_sha256']=previous.profile.reference.saved.identity(decision)
    states=[]
    for frame in previous.STATE_FRAMES:
        value=dict(memory=SimpleNamespace(index={'synthetic':True}),floor={},occupied={},
            residual=SimpleNamespace(pending=None),history=list(range(frame+1)))
        states.append(dict(frame=frame,state_sha256=previous.profile.fingerprint(previous.state_tree(value)),
            retained_observed_state_equal=True))
    return references,{'observed_state_checks':states},tape,calls,models


def test_exact_loop_and_isolated_substitutions_without_module_mutation():
    original=replay.previous.replay
    before=dict(original.__globals__);normalizer=replay.previous.profile.normalize_candidate
    function=replay.isolated_replay()
    assert function.__code__ is original.__code__ and function.__globals__ is not original.__globals__
    assert function.__globals__['FrozenFootprintAnchoredController'] is replay.ScopedFootprintAnchoredController
    assert function.__globals__['ScopedFootprintAnchoredController'] is replay.ScopedBatchedFootprintController
    assert function.__globals__['state_tree'] is replay.normalized_state_tree
    assert function.__globals__['profile'] is not replay.previous.profile
    assert replay.previous.profile.normalize_candidate is normalizer
    assert all(original.__globals__[k] is v for k,v in before.items())


def test_full_incremental_pair_reconstructs_all_frames_and_prior_states(monkeypatch,tmp_path):
    rows,prior,_,calls,models=combined_fixture(monkeypatch,tmp_path)
    result=replay.replay(rows,prior)
    assert calls==[(i,j) for i in range(1428) for j in replay.previous.execution_order(i)]
    assert models[0] is not models[1]
    assert result['frames']==1428 and result['raw_model_forecast_comparisons']==1425
    assert result['observed_state_checks']==prior['observed_state_checks']
    assert result['baseline']=='ScopedFootprintAnchoredController'
    assert result['candidate']=='ScopedBatchedFootprintController'
    assert result['normalized_state_type_paths']==replay.STATE_TYPE_PATHS
    assert result['incremental_batching_comparison'] and result['both_controllers_use_scoped_reuse']
    assert not result['incremental_reuse_comparison'] and not result['imported_module_globals_mutated']
    assert result['no_observation_1428_consumed']


@pytest.mark.parametrize('fault',['receipt','input','metadata','terminal','state','gradient','weight','shared_model','shared_storage'])
def test_combined_loop_rejects_corruption_or_shared_models(monkeypatch,tmp_path,fault):
    rows,prior,_,_,_=combined_fixture(monkeypatch,tmp_path,fault)
    with pytest.raises(ValueError):replay.replay(rows,prior)


@pytest.mark.parametrize('fault',['endpoint','input_hash','original_hash','baseline_hash','prior_state'])
def test_recorded_identity_and_prior_state_cannot_change(monkeypatch,tmp_path,fault):
    rows,prior,tape,_,_=combined_fixture(monkeypatch,tmp_path)
    if fault=='endpoint':tape[1001]['post_sample_index']+=1
    elif fault=='prior_state':prior['observed_state_checks'][-1]['state_sha256']='f'*64
    else:rows[1001][{'input_hash':'public_input_sha256','original_hash':'original_decision_sha256',
        'baseline_hash':'candidate_decision_sha256'}[fault]]='wrong'
    with pytest.raises(ValueError):replay.replay(rows,prior)


def completion_fixture(monkeypatch):
    p=replay.previous;sources={'reviewed.py':'a'*64}
    scope=dict(original_strict_physical_visibility_pass=False,
        original_hard_measurement_failed_frames=[1173],failure_frame_inside_profiled_history=True,
        original_verified_round_trip=False,known_invalid_sensing_retained=True,
        qualified_sensing_prefix_claimed=False,navigation_verified=False)
    monkeypatch.setattr(p.profile,'sensing_scope',lambda:deepcopy(scope))
    rows=[dict(frame=i,execution_order=list(p.execution_order(i)),baseline_controller_s=.2,candidate_controller_s=.1,
        public_input_sha256='a'*64,original_decision_sha256='b'*64,baseline_decision_sha256='c'*64,
        candidate_decision_sha256='d'*64,candidate_normalized_decision_exact=True,
        complete_original_decision_reconstructed=True,public_input_arrays_unchanged=True) for i in range(p.FRAMES)]
    report=dict(frames=p.FRAMES,raw_model_forecast_comparisons=p.FRAMES-3,
        model_state_sha256=p.profile.reference.MODEL_SHA,baseline='FrozenFootprintAnchoredController',
        candidate='ScopedFootprintAnchoredController',state_scope=['memory','mapper.floor','mapper.occupied','residual','history'],
        normalized_state_type_paths=[],model_state_unchanged=True,complete_original_decisions_reconstructed=True,
        complete_normalized_candidate_decisions_exact=True,public_input_arrays_unchanged=True,
        incremental_reuse_comparison=True,alternating_execution_order=True,controller_observe_only_timed=True,
        no_observation_1428_consumed=True,profiling_enabled=False,sensor_acquisition_timed=False,
        isolated_benchmark=False,native_execution=False,real_time_qualified=False,navigation_qualified=False,
        timing_windows=p.timing_summary(rows),observed_state_checks=[dict(frame=i,state_sha256='e'*64,
            retained_observed_state_equal=True) for i in p.STATE_FRAMES])
    result=dict(status='SCOPED_FOOTPRINT_LATE_HISTORY_PAIRED_REPLAY_V1_COMPLETE',
        artifact_sha256={'launch.json':replay.PREVIOUS_LAUNCH_SHA,'comparison.jsonl':'f'*64},
        source_sha256=sources.copy(),sensing_scope=deepcopy(p.profile.sensing_scope()),report=report)
    launch=dict(source_sha256=sources.copy(),profile_result_sha256=replay.PREVIOUS_ARGV[-1])
    return result,launch,rows,sources


def test_exact_complete_predecessor_is_accepted(monkeypatch):
    replay.require_completed(*completion_fixture(monkeypatch))


@pytest.mark.parametrize('fault',['status','artifact','source','profile_link','frames','forecast_count','model',
    'state_scope','type_normalization','baseline','candidate','missing_row','row_order','failed_row','hash',
    'timing','missing_state','state_flag','weights','qualification','visibility'])
def test_incomplete_or_changed_predecessor_is_rejected(monkeypatch,fault):
    result,launch,rows,sources=completion_fixture(monkeypatch);report=result['report']
    if fault=='status':result['status']='RUNNING'
    elif fault=='artifact':result['artifact_sha256'].pop('comparison.jsonl')
    elif fault=='source':result['source_sha256']['reviewed.py']='f'*64
    elif fault=='profile_link':launch['profile_result_sha256']='f'*64
    elif fault=='frames':report['frames']-=1
    elif fault=='forecast_count':report['raw_model_forecast_comparisons']-=1
    elif fault=='model':report['model_state_sha256']='f'*64
    elif fault=='state_scope':report['state_scope'].pop()
    elif fault=='type_normalization':report['normalized_state_type_paths']=['arbitrary']
    elif fault in ('baseline','candidate'):report[fault]='different'
    elif fault=='missing_row':rows.pop()
    elif fault=='row_order':rows[1001]['frame']=1000
    elif fault=='failed_row':rows[1001]['candidate_normalized_decision_exact']=False
    elif fault=='hash':rows[1001]['candidate_decision_sha256']='wrong'
    elif fault=='timing':rows[1001]['candidate_controller_s']=1.
    elif fault=='missing_state':report['observed_state_checks'].pop()
    elif fault=='state_flag':report['observed_state_checks'][-1]['retained_observed_state_equal']=False
    elif fault=='weights':report['model_state_unchanged']=False
    elif fault=='qualification':report['real_time_qualified']=True
    else:result['sensing_scope']['original_strict_physical_visibility_pass']=True
    with pytest.raises(ValueError):replay.require_completed(result,launch,rows,sources)


@pytest.mark.parametrize('mode',['live','reused','gone'])
def test_original_scoped_owner_must_end(monkeypatch,mode):
    monkeypatch.setattr(replay.Path,'read_text',lambda *a,**k:replay.previous.BOOT)
    def process(pid):
        assert pid==replay.PREVIOUS_PID
        if mode=='gone':raise replay.psutil.NoSuchProcess(pid)
        return SimpleNamespace(create_time=lambda:replay.PREVIOUS_CREATED+(mode=='reused'),cmdline=lambda:replay.PREVIOUS_ARGV)
    monkeypatch.setattr(replay.psutil,'Process',process)
    if mode=='gone':replay.previous_owner_ended()
    else:
        with pytest.raises(ValueError):replay.previous_owner_ended()


def test_source_preflight_does_not_admit_inputs_or_execute(monkeypatch,tmp_path):
    for key,value in dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0').items():
        monkeypatch.setenv(key,value)
    monkeypatch.setattr(replay,'OUTPUT',tmp_path/'absent')
    monkeypatch.setattr(replay,'validate_root',lambda *a,**k:None)
    monkeypatch.setattr(replay,'prepared_sources',lambda:{})
    monkeypatch.setattr(replay.previous.profile.reference,'hardware',lambda:dict(
        memory_available_bytes=64*1024**3,artifact_free_bytes=41*1024**3,physical_cpus=4))
    def forbidden(*a,**k):pytest.fail('runtime during source preparation')
    for name in ('completed_previous','create_output','replay'):monkeypatch.setattr(replay,name,forbidden)
    monkeypatch.setattr(replay.previous.profile.reference,'admit_worker',forbidden)
    monkeypatch.setattr(sys,'argv',[replay.SOURCE,'--source-preflight-only'])
    replay.main()
    assert not replay.OUTPUT.exists()


def test_final_input_change_preserves_failure_and_prevents_second_attempt(monkeypatch,tmp_path):
    for key,value in dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0').items():
        monkeypatch.setenv(key,value)
    root=tmp_path/'attempt';monkeypatch.setattr(replay,'OUTPUT',root)
    monkeypatch.setattr(replay,'validate_root',lambda *a,**k:None)
    monkeypatch.setattr(replay,'prepared_sources',lambda:{})
    monkeypatch.setattr(replay,'previous_owner_ended',lambda:None)
    monkeypatch.setattr(replay,'create_output',lambda p:p.mkdir())
    monkeypatch.setattr(replay,'completed_previous',lambda *a:({'sensing_scope':{},'report':{}},[]))
    monkeypatch.setattr(replay.previous.profile.reference,'hardware',lambda:dict(
        memory_available_bytes=64*1024**3,artifact_free_bytes=41*1024**3,physical_cpus=4))
    admissions=iter([{'identity':'original'},{'identity':'changed'}])
    monkeypatch.setattr(replay.previous.profile.reference,'admit_worker',lambda *a:next(admissions))
    def run(*a):
        (root/'comparison.jsonl').write_text('synthetic completed comparison\n')
        return {}
    monkeypatch.setattr(replay,'replay',run)
    monkeypatch.setattr(sys,'argv',[replay.SOURCE,'--scoped-result-sha256','a'*64])
    with pytest.raises(ValueError,match='input admission or scoped reference changed'):replay.main()
    assert (root/'comparison.jsonl').read_text()=='synthetic completed comparison\n'
    assert json.loads((root/'failure.json').read_text())['status']=='TERMINAL_SCOPED_BATCHED_FOOTPRINT_REPLAY_FAILURE'
    assert not (root/'result.json').exists()
    with pytest.raises(ValueError,match='no retry or resume'):replay.main()
