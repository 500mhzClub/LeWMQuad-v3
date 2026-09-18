from copy import deepcopy
import hashlib
from itertools import islice
import sys
import pytest

from scripts import verify_go2_reused_floor_mesh_prefix_v1 as checker


def population():
    replay=checker.replay; identity=replay.profile.reference.saved.identity
    rows=[];saved=[];profiled=[];frozen=[];tape=[]
    for i in range(405):
        d=dict(controller='residual_anchored_continuation_controller_v1',terminal=None,
            requested_command=[0.,0.,0.] if i<3 else [.2,0.,0.],
            new_selection=None if i<3 else {'prediction':[float(i)],'nested':{replay.FLAG:False}})
        old=identity(d);public=hashlib.sha256(str(i).encode()).hexdigest()
        new=d|{'controller':replay.CONTROLLER,replay.FLAG:True,replay.frozen.FLAG:True}
        saved.append({'tick':i,'decision':d})
        row=dict(frame=i,original_decision_sha256=old,candidate_decision_sha256=identity(new),
            public_input_sha256=public,complete_original_decision_reconstructed=True,
            candidate_normalized_decision_exact=True,public_input_arrays_unchanged=True,
            execution_order=list(replay.preceding.execution_order(i)),
            original_controller_s=.8,candidate_controller_s=.6)
        rows.append(row)
        profiled.append(dict(frame=i,original_decision_sha256=old,public_input_sha256=public))
        frozen.append(dict(profiled[-1]))
        tape.append(dict(tick=i,completed=True,pre_sample_index=749+50*i,
            post_sample_index=799+50*i,requested_command=d['requested_command']))
    return rows,saved,profiled,frozen,tape,replay.preceding.timing_summary(rows)


def test_complete_population_and_windows_reconstruct_without_reading_observation_405():
    rows,saved,profiled,frozen,tape,timings=population();consumed=[]
    def original():
        for row in saved:
            consumed.append(row['tick']);yield row
        raise AssertionError('observation 405 consumed')
    actual=checker.check_rows(rows,islice(original(),405),profiled,frozen,tape,timings)
    assert consumed==list(range(405))
    assert actual['candidate_expected_decision_hashes_checked']==405 and actual['forecast_count']==402
    assert actual['timing_windows']==timings
    assert actual['timing_windows']['post_warmup_prefix']['candidate_over_100ms']==402


@pytest.mark.parametrize('fault',['missing','saved_missing','saved_extra','order','command','endpoint',
    'candidate','missing_frozen_flag','public','state_flag','terminal','forecast','timing','timing_report'])
def test_corrupted_comparisons_or_unsupported_claims_are_rejected(fault):
    rows,saved,profiled,frozen,tape,timings=population()
    if fault=='missing':rows.pop()
    elif fault=='saved_missing':saved.pop()
    elif fault=='saved_extra':saved.append(saved[-1])
    elif fault=='order':rows[3]['execution_order']=[0,1]
    elif fault=='command':tape[3]['requested_command']=[0.,0.,.45]
    elif fault=='endpoint':tape[3]['post_sample_index']+=1
    elif fault=='candidate':rows[3]['candidate_decision_sha256']='a'*64
    elif fault=='missing_frozen_flag':
        expected=saved[3]['decision']|{'controller':checker.replay.CONTROLLER,checker.replay.FLAG:True}
        rows[3]['candidate_decision_sha256']=checker.replay.profile.reference.saved.identity(expected)
    elif fault=='public':frozen[3]['public_input_sha256']='a'*64
    elif fault=='state_flag':rows[3]['public_input_arrays_unchanged']=1
    elif fault=='terminal':saved[3]['decision']['terminal']='FAILURE'
    elif fault=='forecast':saved[3]['decision']['new_selection'].pop('prediction')
    elif fault=='timing':rows[3]['candidate_controller_s']=float('nan')
    else:timings['early_navigation']['candidate_total_s']=0.
    with pytest.raises(ValueError):checker.check_rows(rows,saved,profiled,frozen,tape,timings)


def reports():
    r=checker.replay
    launch=dict(source_sha256={},optimized_profile_result_sha256=r.PROFILE_RESULT_SHA,
        optimized_profile_launch_sha256=r.PROFILE_LAUNCH_SHA,
        paired_frozen_footprint_result_sha256=r.optimized_profile.COMPLETED_SHA,
        frames=405,state_frames=[3,12,395,404],model_state_sha256=r.profile.reference.MODEL_SHA,
        native_execution=False,model_training=False,profiling_enabled=False,
        imported_module_globals_mutated=False,invocation_frozen_footprint_receipts=True,
        observation_local_floor_mesh_reuse=True,normalized_state_type_paths=[],
        comparison='original controller versus combined frozen-footprint and mesh-reuse controller',
        isolated_incremental_mesh_speedup_claimed=False)
    report=dict(frames=405,raw_model_forecast_comparisons=402,
        model_state_sha256=r.profile.reference.MODEL_SHA,model_state_unchanged=True,
        complete_original_decisions_reconstructed=True,complete_normalized_candidate_decisions_exact=True,
        public_input_arrays_unchanged=True,alternating_execution_order=True,profiling_enabled=False,
        controller_observe_only_timed=True,sensor_acquisition_timed=False,isolated_benchmark=False,
        no_observation_405_consumed=True,native_execution=False,real_time_qualified=False,
        navigation_qualified=False,observed_state_checks=[dict(frame=i,state_sha256=str(i),
        complete_retained_observed_state_equal=True) for i in r.STATE_FRAMES])
    result=dict(status='REUSED_FLOOR_MESH_PREFIX_V1_COMPLETE',source_sha256={},
        native_execution=False,goal_achieved=False,isolated_incremental_mesh_speedup_claimed=False,
        optimized_profile_result_sha256=r.PROFILE_RESULT_SHA,wall_s=20.,report=report)
    reference={'report':{'observed_state_checks':deepcopy(report['observed_state_checks'])}}
    return result,launch,reference


def test_complete_report_scope_and_state_receipts_are_admitted():
    checker.require_report(*reports())


@pytest.mark.parametrize('fault',['incomplete','claimed_incremental_speedup','state_hash','state_missing',
    'state_normalization','model','source','profiled','native','invalid_wall','bool_count'])
def test_incomplete_report_or_broader_claims_are_rejected(fault):
    result,launch,reference=reports()
    if fault=='incomplete':result['status']='INCOMPLETE'
    elif fault=='claimed_incremental_speedup':result['isolated_incremental_mesh_speedup_claimed']=True
    elif fault=='state_hash':result['report']['observed_state_checks'][0]['state_sha256']='changed'
    elif fault=='state_missing':result['report']['observed_state_checks'].pop()
    elif fault=='state_normalization':launch['normalized_state_type_paths']=['memory']
    elif fault=='model':result['report']['model_state_sha256']='a'*64
    elif fault=='source':launch['source_sha256']['different']='a'*64
    elif fault=='profiled':launch['profiling_enabled']=True
    elif fault=='native':result['native_execution']=True
    elif fault=='invalid_wall':result['wall_s']=float('nan')
    else:result['report']['frames']=True
    with pytest.raises(ValueError):checker.require_report(result,launch,reference)


def test_source_preflight_does_not_open_replay_outputs_or_write_verification(monkeypatch,capsys):
    def forbidden(*args,**kwargs):raise AssertionError('runtime work during source preflight')
    monkeypatch.setattr(checker,'prepared_sources',lambda:({},{}))
    monkeypatch.setattr(checker,'verified_result',forbidden)
    monkeypatch.setattr(checker,'write_json',forbidden)
    monkeypatch.setattr(sys,'argv',[checker.SOURCE,'--source-preflight-only'])
    checker.main()
    assert 'REUSED_FLOOR_MESH_VERIFICATION_SOURCE_PREFLIGHT_PASS 0' in capsys.readouterr().out


def test_incomplete_jsonl_is_rejected(monkeypatch,tmp_path):
    f=tmp_path/'comparison.jsonl';f.write_text('{"frame":0}')
    monkeypatch.setattr(checker,'artifact_path',lambda *args:f)
    with pytest.raises(ValueError,match='newline'):checker.rows_at(tmp_path)
