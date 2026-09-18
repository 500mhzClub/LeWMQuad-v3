"""Incomplete timing populations and unsupported completion claims must fail."""
from copy import deepcopy
import json
from types import SimpleNamespace
import pytest
from scripts import verify_go2_tiled_density_progressive_floor_profile_completion_v2 as check


def fixture():
    rows=[];summaries={};windows={}
    for frame in range(1428):
        window=next((n for n,(a,b) in check.run.original.WINDOWS.items() if a<=frame<=b),None)
        rows.append(dict(frame=frame,profiled_window=window,controller_wall_s=.5))
    for name,(first,last) in check.run.original.WINDOWS.items():
        summary=dict(total_exclusive_profiled_s=4.,functions=[{'function':'observed','time':3.}],
            modules_by_exclusive_time=[{'module':'observed','time':4.}])
        summaries[name]=summary
        windows[name]=dict(observations=[dict(frame=i,action='hold' if name=='repeated_hold' else 'forward',
            controller_wall_s_with_profiling=.5) for i in range(first,last+1)],
            total_exclusive_profiled_s=4.,top_functions_by_cumulative_time=deepcopy(summary['functions']),
            top_modules_by_exclusive_time=deepcopy(summary['modules_by_exclusive_time']))
    report=dict(frames=1428,raw_model_forecast_comparisons=1425,model_state_sha256=check.run.original.reference.MODEL_SHA,
        last_replayed_observation=1427,controller='TiledDensityProgressiveFloorController',windows=windows,
        state_size_snapshots={'unverified':True},retained_state_identity_established=False,sensor_acquisition_profiled=False,
        profiler_overhead_removed=False,isolated_benchmark=False,speedup_established=False,native_execution=False,
        policy_changed=False,real_time_qualified=False,navigation_qualified=False,imported_module_globals_mutated=False,
        complete_original_decisions_reconstructed=True,model_state_unchanged=True,controller_observe_only_profiled=True,
        no_observation_1428_consumed=True,invocation_frozen_footprint_receipts=True,normalization_outside_profiled_region=True,
        complete_normalized_candidate_decisions_exact=True,tiled_density_progressive_floor_controller_profiled=True,
        all_profiled_decisions_equal_completed_tiled_replay=True)
    return rows,report,summaries


def test_complete_fixed_timing_windows_reconstruct_without_promoting_scope():
    rows,report,summaries=fixture()
    result=check.reconstruct_report(rows,report,summaries)
    assert result==report['windows']
    assert report['speedup_established'] is False and report['retained_state_identity_established'] is False


@pytest.mark.parametrize('fault',['short','extra','order','bool_frame','summary_missing','window_missing','wrong_window',
    'nan','zero','negative','bool_time','population','timing','hold_action','total','functions','modules',
    'model','forecasts','controller','speedup','native','state_claim','extra_claim','last_frame'])
def test_incomplete_or_changed_profile_evidence_is_rejected(fault):
    rows,report,summaries=fixture();window=report['windows']['repeated_hold']
    if fault=='short':rows.pop()
    if fault=='extra':rows.append(deepcopy(rows[-1]))
    if fault=='order':rows[2]['frame']=1
    if fault=='bool_frame':rows[0]['frame']=False
    if fault=='summary_missing':summaries.pop('early_navigation')
    if fault=='window_missing':report['windows'].pop('early_navigation')
    if fault=='wrong_window':rows[0]['profiled_window']='early_navigation'
    if fault in ('nan','zero','negative','bool_time'):
        rows[0]['controller_wall_s']={'nan':float('nan'),'zero':0.,'negative':-1.,'bool_time':True}[fault]
    if fault=='population':window['observations'].pop()
    if fault=='timing':window['observations'][0]['controller_wall_s_with_profiling']=.6
    if fault=='hold_action':window['observations'][0]['action']='forward'
    if fault=='total':window['total_exclusive_profiled_s']=5.
    if fault=='functions':window['top_functions_by_cumulative_time']=[]
    if fault=='modules':window['top_modules_by_exclusive_time']=[]
    if fault=='model':report['model_state_sha256']='changed'
    if fault=='forecasts':report['raw_model_forecast_comparisons']=1424
    if fault=='controller':report['controller']='PackedFusedScopedController'
    if fault=='speedup':report['speedup_established']=True
    if fault=='native':report['native_execution']=True
    if fault=='state_claim':report['retained_state_identity_established']=True
    if fault=='extra_claim':report['hardware_qualified']=True
    if fault=='last_frame':report['last_replayed_observation']=1428
    with pytest.raises(ValueError):check.reconstruct_report(rows,report,summaries)


@pytest.mark.parametrize('value',[None,True,'','a'*63,'g'*64])
def test_explicit_result_identity_required_before_io(monkeypatch,value):
    monkeypatch.setattr(check,'verify',lambda *a:pytest.fail('invalid hash must fail before source IO'))
    with pytest.raises(ValueError,match='SHA-256'):check.verify_completed(value, 'b'*64)


def test_live_owner_rejects_before_result_or_profile_access(monkeypatch,tmp_path):
    execution=dict(boot_id='boot',owner={'pid':123})
    (tmp_path/'execution.json').write_text(json.dumps(execution))
    monkeypatch.setattr(check,'ROOT',tmp_path);monkeypatch.setattr(check,'EXECUTION','execution.json')
    monkeypatch.setattr(check,'Path',lambda *a:SimpleNamespace(read_text=lambda:'boot'))
    monkeypatch.setattr(check,'verify',lambda *a:None)
    monkeypatch.setattr(check,'owner_live',lambda *a:True)
    monkeypatch.setattr(check,'read_json',lambda *a:pytest.fail('live output must not be read'))
    monkeypatch.setattr(check.run,'prepared_sources',lambda:pytest.fail('completion must wait for owner'))
    with pytest.raises(ValueError,match='owner still live'):check.verify_completed('a'*64, 'b'*64)


@pytest.mark.parametrize('value',[None,True,'','a'*63,'g'*64])
def test_execution_identity_required_before_io(monkeypatch,value):
    monkeypatch.setattr(check,'verify',lambda *a:pytest.fail('invalid execution binding must fail first'))
    with pytest.raises(ValueError,match='SHA-256'):check.verify_completed('a'*64,value)


def test_existing_completion_receipt_rejected_before_reading_profile(monkeypatch,tmp_path):
    import sys
    monkeypatch.setattr(check,'OUTPUT',tmp_path)
    monkeypatch.setattr(check,'verify_completed',lambda *a:pytest.fail('existing receipt must reject first'))
    monkeypatch.setattr(sys,'argv',['check','--result-sha256','a'*64,'--execution-sha256','b'*64])
    with pytest.raises(ValueError,match='exclusive'):check.main()
