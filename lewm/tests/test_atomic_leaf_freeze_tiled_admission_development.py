"""Reconstruct actual predecessor reports and reject altered evidence before replay."""
from copy import deepcopy
import json
import pytest
from scripts import replay_go2_atomic_leaf_freeze_tiled_late_history_v1 as run
from lewm.tests.test_tiled_density_progressive_floor_controller_completion_development import histories


def fixture(monkeypatch,tmp_path):
    rows,prior_rows=histories()
    sources={'source.py':'a'*64};admission={'original_raw_and_model':'unchanged'}
    prior_report=dict(incremental_progressive_patch_batching_comparison=True,
        both_controllers_use_density_routed_floor_registration_and_mapping=True,
        observed_state_checks=[dict(frame=i,state_sha256=str(i)) for i in (3,12,395,404,1173,1418,1427)],
        original_sensing_failure=1173,navigation_qualified=False,model_state_unchanged=True)
    timing=run.previous_check.check_rows(rows,prior_rows)
    scope={'original_strict_failure':1173}
    prior=dict(report=prior_report,sensing_scope=scope.copy())
    launch=dict(source_sha256=sources.copy(),input_admission=admission.copy())
    result=dict(status='TILED_DENSITY_PROGRESSIVE_FLOOR_LATE_HISTORY_V1_COMPLETE',source_sha256=sources.copy(),
        artifact_sha256={'launch.json':run.PREVIOUS_LAUNCH_SHA,'comparison.jsonl':'c'*64},
        native_execution=False,goal_achieved=False,sensing_scope=scope.copy(),
        report=run.previous_check.expected_report(prior_report,timing))
    raw={'original':'raw launch'};calls=[]
    monkeypatch.setattr(run.previous,'OUTPUT',tmp_path)
    monkeypatch.setattr(run,'previous_owner_ended',lambda:calls.append('owner'))
    monkeypatch.setattr(run,'verify',lambda value:calls.append(('sources',deepcopy(value))))
    monkeypatch.setattr(run,'verify_artifacts',lambda root,ids:calls.append(('artifacts',root,ids.copy())))
    monkeypatch.setattr(run,'read_json',lambda root,name:{'launch.json':launch,'result.json':result}[name])
    monkeypatch.setattr(run.previous,'admit_completed',lambda source:(None,prior,deepcopy(launch),prior_rows,raw))
    monkeypatch.setattr(run.profile.profile,'bound_profile_inputs',lambda received,source:admission.copy())
    (tmp_path/'comparison.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return sources,result,launch,rows,prior,calls,raw


def test_complete_predecessor_is_reconstructed_with_every_row_and_negative_scope(monkeypatch,tmp_path):
    sources,result,launch,rows,prior,calls,raw=fixture(monkeypatch,tmp_path)
    fixed,actual,actual_launch,actual_rows,actual_raw=run.admit_completed(sources)
    assert calls[0]=='owner'
    assert actual is result and actual_launch is launch and actual_raw is raw and actual_rows==rows
    assert len(actual_rows)==1428 and actual['report']['timing_windows']['all_navigation']['candidate_over_100ms']==1425
    assert actual['report']['observed_state_checks']==prior['report']['observed_state_checks']
    assert actual['report']['original_sensing_failure']==1173 and actual['sensing_scope']==prior['sensing_scope']
    assert calls[-1]==('artifacts',tmp_path,result['artifact_sha256']|fixed)


@pytest.mark.parametrize('fault',['status','source','launch_source','artifacts','launch_hash','native','goal',
    'state','report_extra','sensing','admission','input_hash','baseline_hash','row_count','timing','claim','failure_file'])
def test_corrupted_predecessor_cannot_be_used(monkeypatch,tmp_path,fault):
    sources,result,launch,rows,prior,calls,raw=fixture(monkeypatch,tmp_path)
    if fault=='status':result['status']='OTHER'
    elif fault=='source':result['source_sha256']['source.py']='b'*64
    elif fault=='launch_source':launch['source_sha256']['source.py']='b'*64
    elif fault=='artifacts':result['artifact_sha256']['extra']='d'*64
    elif fault=='launch_hash':result['artifact_sha256']['launch.json']='d'*64
    elif fault=='native':result['native_execution']=True
    elif fault=='goal':result['goal_achieved']=True
    elif fault=='state':result['report']['observed_state_checks'][-1]['state_sha256']='changed'
    elif fault=='report_extra':result['report']['unreviewed_claim']=True
    elif fault=='sensing':result['sensing_scope']['original_strict_failure']=None
    elif fault=='admission':launch['input_admission']['original_raw_and_model']='changed'
    elif fault=='input_hash':rows[800]['public_input_sha256']='d'*64
    elif fault=='baseline_hash':rows[800]['baseline_decision_sha256']='e'*64
    elif fault=='row_count':rows.pop()
    elif fault=='timing':rows[800]['candidate_controller_s']=-1.
    elif fault=='claim':rows[800]['candidate_normalized_decision_exact']=1
    elif fault=='failure_file':(tmp_path/'failure.json').write_text('{}')
    (tmp_path/'comparison.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError):run.admit_completed(sources)


def test_live_predecessor_rejected_before_artifact_access(monkeypatch,tmp_path):
    sources,_,_,_,_,calls,_=fixture(monkeypatch,tmp_path)
    def live():raise ValueError('still live')
    monkeypatch.setattr(run,'previous_owner_ended',live)
    with pytest.raises(ValueError,match='still live'):run.admit_completed(sources)
    assert calls==[]


@pytest.mark.parametrize('field',['WITNESS_SHA','PREVIOUS_RESULT_SHA'])
def test_unbound_result_cannot_reach_source_or_artifact_reads(monkeypatch,field):
    monkeypatch.setattr(run,'WITNESS_SHA','a'*64)
    monkeypatch.setattr(run,'PREVIOUS_RESULT_SHA','b'*64)
    monkeypatch.setattr(run,field,'pending')
    monkeypatch.setattr(run,'verify',lambda *a:pytest.fail('unbound predecessor must fail first'))
    with pytest.raises(ValueError,match='actual completed'):run.prepared_sources()


def test_launcher_uses_the_separately_checked_harness_without_a_second_output():
    assert run.replay is run.harness.replay
    assert run.OUTPUT is run.harness.OUTPUT
    assert run.original is run.harness.original
