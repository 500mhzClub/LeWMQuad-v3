"""Completed-reference and isolated profiling checks; no real controller run."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from scripts import profile_go2_scoped_batched_footprint_late_history_v1 as p


def fixture():
    sources={'synthetic.py':'a'*64}
    prior=[dict(frame=i,public_input_sha256='b'*64,original_decision_sha256='c'*64,
        candidate_decision_sha256='d'*64) for i in range(1428)]
    rows=[dict(r,baseline_decision_sha256=r['candidate_decision_sha256'],candidate_decision_sha256='e'*64,
        execution_order=list(p.paired.previous.execution_order(i)),baseline_controller_s=.2,candidate_controller_s=.15,
        candidate_normalized_decision_exact=True,complete_original_decision_reconstructed=True,
        public_input_arrays_unchanged=True) for i,r in enumerate(prior)]
    states=[dict(frame=i,state_sha256='f'*64,retained_observed_state_equal=True) for i in p.paired.previous.STATE_FRAMES]
    report=dict(frames=1428,raw_model_forecast_comparisons=1425,model_state_sha256=p.original.reference.MODEL_SHA,
        baseline='ScopedFootprintAnchoredController',candidate='ScopedBatchedFootprintController',
        normalized_state_type_paths=p.paired.STATE_TYPE_PATHS,
        state_scope=['memory','mapper.floor','mapper.occupied','residual','history'],observed_state_checks=states,
        timing_windows=p.paired.previous.timing_summary(rows))
    for k in ('complete_original_decisions_reconstructed','complete_normalized_candidate_decisions_exact',
        'public_input_arrays_unchanged','model_state_unchanged','alternating_execution_order',
        'controller_observe_only_timed','no_observation_1428_consumed','incremental_batching_comparison',
        'both_controllers_use_scoped_reuse'):report[k]=True
    for k in ('incremental_reuse_comparison','imported_module_globals_mutated','profiling_enabled',
        'sensor_acquisition_timed','isolated_benchmark','native_execution','real_time_qualified','navigation_qualified'):report[k]=False
    result=dict(status='SCOPED_BATCHED_FOOTPRINT_LATE_HISTORY_REPLAY_V1_COMPLETE',source_sha256=sources,
        artifact_sha256={'launch.json':p.PAIRED_LAUNCH,'comparison.jsonl':'1'*64},report=report,
        sensing_scope={'synthetic_negative':True},native_execution=False,goal_achieved=False)
    launch=dict(source_sha256=sources,scoped_result_sha256=p.SCOPED_SHA)
    preceding=dict(sensing_scope=deepcopy(result['sensing_scope']),report={'observed_state_checks':deepcopy(states)})
    return result,launch,rows,preceding,prior,sources


def test_complete_combined_reference_accepted(): p.require_complete(*fixture())


@pytest.mark.parametrize('fault',['partial','timing','input','baseline','state','scope','policy','normalization','source'])
def test_changed_or_incomplete_reference_rejected(fault):
    args=fixture();result,launch,rows,preceding,prior,sources=args
    if fault=='partial':rows.pop()
    elif fault=='timing':rows[-1]['candidate_controller_s']*=2
    elif fault=='input':rows[-1]['public_input_sha256']='0'*64
    elif fault=='baseline':rows[-1]['baseline_decision_sha256']='0'*64
    elif fault=='state':result['report']['observed_state_checks'][-1]['state_sha256']='0'*64
    elif fault=='scope':result['sensing_scope']['synthetic_negative']=False
    elif fault=='policy':result['report']['candidate']='another_controller'
    elif fault=='normalization':result['report']['normalized_state_type_paths']=[]
    else:sources=dict(sources,**{'synthetic.py':'0'*64});args=(*args[:-1],sources)
    with pytest.raises(ValueError):p.require_complete(*args)


def test_profiler_body_only_changes_declared_globals():
    original=p.original.replay;before=dict(original.__globals__)
    clone=p.isolated_replay()
    assert clone.__code__ is original.__code__ and clone.__globals__ is not original.__globals__
    assert clone.__globals__['FrozenFootprintAnchoredController'] is p.paired.ScopedBatchedFootprintController
    assert clone.__globals__['normalize_candidate'] is p.paired.normalize_candidate
    assert clone.__globals__['OUTPUT']==p.OUTPUT
    for k,v in before.items():
        assert original.__globals__[k] is v
        if k not in ('FrozenFootprintAnchoredController','normalize_candidate','OUTPUT','print'):
            assert clone.__globals__[k] is v


@pytest.mark.parametrize('field',['candidate_decision_sha256','original_decision_sha256','public_input_sha256'])
def test_profiling_must_preserve_all_completed_decision_bindings(field):
    rows=fixture()[2];expected=deepcopy(rows)
    p.compare_profile_rows(rows,expected)
    rows[-1][field]='0'*64
    with pytest.raises(ValueError,match='profiling changed'):p.compare_profile_rows(rows,expected)


@pytest.mark.parametrize('state',['live','reused','ended'])
def test_prior_full_replay_slot_requires_original_owner_ended(monkeypatch,state):
    def process(pid):
        assert pid==p.PAIRED_OWNER['pid']
        if state=='ended':raise p.psutil.NoSuchProcess(pid)
        return SimpleNamespace(create_time=lambda:p.PAIRED_OWNER['created']+(state=='reused'),
            cmdline=lambda:p.PAIRED_OWNER['command'])
    monkeypatch.setattr(p.psutil,'Process',process)
    if state=='ended':p.paired_owner_ended()
    else:
        with pytest.raises(ValueError):p.paired_owner_ended()


def test_profile_inputs_rehash_raw_bindings_without_claiming_new_full_ancestry(monkeypatch):
    calls=[];native=p.original.reference.original;name=p.original.reference.CASE[0]
    ids={'launch.json':'a'*64,'synthetic_raw':'b'*64}
    admission=dict(original_worker_terminal_sha256=p.original.WORKER_SHA,original_case=name,
        original_worker_complete_and_raw_audited=True,original_full_input_verifier_reexecuted=True,
        original_artifact_sha256=ids)
    launch={'input_admission':admission};sources={'synthetic.py':'c'*64}
    documents={'launch.json':{'source_sha256':sources},name+'_worker_terminal.json':dict(
        model_state_sha256=p.original.reference.MODEL_SHA,artifact_sha256={'synthetic_raw':'b'*64}),name+'_audit.json':{}}
    monkeypatch.setattr(p,'verify_artifacts',lambda root,bindings:calls.append(('raw',deepcopy(bindings))))
    monkeypatch.setattr(p,'verify',lambda source:calls.append(('sources',source)))
    monkeypatch.setattr(p,'read_json',lambda root,name:deepcopy(documents[name]))
    monkeypatch.setattr(p.original.reference,'require_case',lambda *a:calls.append(('case',a[0])))
    monkeypatch.setattr(native,'verify_inputs',lambda launch,*,full:calls.append(('input_verifier',full)))
    monkeypatch.setattr(p.original,'sensing_scope',lambda:{'negative_preserved':True})
    result=p.bound_profile_inputs(launch,sources)
    assert ('raw',ids) in calls and ('input_verifier',False) in calls
    assert result['prior_completed_replay_full_input_admission_reused']
    assert result['complete_bound_raw_worker_artifacts_rehashed'] and not result['full_training_ancestry_reexecuted']
    admission['original_full_input_verifier_reexecuted']=False
    with pytest.raises(ValueError):p.bound_profile_inputs(launch,sources)
