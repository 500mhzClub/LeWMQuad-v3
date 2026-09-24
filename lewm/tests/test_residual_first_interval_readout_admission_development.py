"""Require completed paired raw evidence without discarding scientific failures."""
from copy import deepcopy
import pytest
from scripts import read_go2_residual_first_interval_maze_pilot_v1 as runner


def fixture():
    outcomes=dict(verified_round_trip=False,native_evaluation={'arrival_windows':[]},
        strict_physical_visibility_pass=False,hard_measurement_failed_frames=[18],renderer_capture_audit={})
    audit=dict(layout_index=2,raw_sensor_reconstruction_pass=True,additional_auxiliary_rgb_reconstructed=True,
        raw_model_command_replay_pass=True,raw_command_audit_pass=True,model_state_unchanged=True,**deepcopy(outcomes))
    old={k:'same' for k in runner.MATCHED_KEYS}
    old.update(model_state_sha256=runner.MODEL_STATE,planned_cases=[list(runner.LEARNED_CASE)],
        implementation_class='MeasuredFloorTransportController',correction_admission={'fit':'same'},
        scene_specifications=[{'layout':i} for i in (1,2,3)],public_missions=[{'goal':i} for i in (1,2,3)])
    prior=dict(case=runner.LEARNED_CASE[0],layout_index=2,
        status='INDEPENDENT_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED',**deepcopy(outcomes))
    learned=dict(status='INDEPENDENT_FLOOR_TRANSPORT_MAZES_V1_COMPLETE',all_fixed_cases_executed=True,
        conditions=[{'layout_index':1},prior,{'layout_index':3}])
    prefix=dict(common_prefix_frames=464,first_intervention_frame=463,physical_prefix_samples=23900,
        physical_and_public_prefix_exact=True,shared_observed_mission_and_residual_state_exact=True,
        all_preintervention_requested_commands_exact=True,complete_candidate_decisions_match_prospective_prefix=True,
        raw_model_forecast_comparisons=461,all_compared_raw_model_forecasts_exact=True,
        original_intervention_command=[0.,0.,0.],candidate_intervention_command=[.16,0.,-.45],
        following_physical_outcomes_compared=False,unexecuted_outcomes_inferred=False)
    record=dict(case=runner.CASE[0],layout_index=2,status='RESIDUAL_FIRST_INTERVAL_COLLECTED_AND_RAW_AUDITED',
        model_state_unchanged=True,prefix_comparison=prefix,**deepcopy(outcomes))
    current=dict(status='RESIDUAL_FIRST_INTERVAL_MAZE_PILOT_V1_COMPLETE',conditions=[record],
        learned_cohort_result_sha256=runner.LEARNED_SHA)
    launch={k:old[k] for k in runner.MATCHED_KEYS}
    launch.update(planned_case=list(runner.CASE),implementation_class='ResidualFirstIntervalController',
        prefix_report={'model_state_sha256':runner.MODEL_STATE},correction_admission=deepcopy(old['correction_admission']),
        scene_specification=deepcopy(old['scene_specifications'][1]),public_mission=deepcopy(old['public_missions'][1]))
    return current,launch,learned,old,deepcopy(audit),audit


@pytest.mark.parametrize('fault',[None,'incomplete','model','mission','correction','raw_audit','outcome','prefix','setting'])
def test_complete_matching_results_required_and_negative_retained(fault):
    args=fixture(); current,launch,learned,old,audit,old_audit=args
    if fault=='incomplete': current['status']='TERMINAL_RESIDUAL_FIRST_INTERVAL_NATIVE_FAILURE'
    elif fault=='model': launch['prefix_report']['model_state_sha256']='0'*64
    elif fault=='mission': launch['public_mission']['goal']=7
    elif fault=='correction': launch['correction_admission']['fit']='changed'
    elif fault=='raw_audit': audit['raw_model_command_replay_pass']=False
    elif fault=='outcome': audit['verified_round_trip']=True
    elif fault=='prefix': current['conditions'][0]['prefix_comparison']['physical_prefix_samples']=23899
    elif fault=='setting': launch['navigation_ticks']='changed'
    before=deepcopy(args)
    if fault is None:
        result=runner.admit_results(*args)
        assert result['original_outcomes_unchanged'] and result['paired_failure_retained']
    else:
        with pytest.raises(ValueError): runner.admit_results(*args)
    assert args==before


def test_readout_launch_retains_required_verifier_bindings(monkeypatch,tmp_path):
    current,launch,learned,old,audit,old_audit=fixture()
    current['artifact_sha256']={}; current['source_sha256']={'synthetic_source':'a'*64}
    launch['learned_artifact_sha256']={'result.json':runner.LEARNED_SHA}
    monkeypatch.setattr(runner,'OUTPUT',tmp_path/'output')
    monkeypatch.setattr('sys.argv',['readout','--native-result-sha256','b'*64])
    monkeypatch.setattr(runner,'validate_root',lambda *a,**k:None)
    monkeypatch.setattr(runner,'verify_artifacts',lambda *a:None)
    monkeypatch.setattr(runner,'verify_native',lambda *a:None)
    def read(root,name):
        if root==runner.INPUT: return {'result.json':current,'launch.json':launch,runner.CASE[0]+'_audit.json':audit}[name]
        return {'result.json':learned,'launch.json':old,runner.LEARNED_CASE[0]+'_audit.json':old_audit}[name]
    monkeypatch.setattr(runner,'read_json',read)
    monkeypatch.setattr(runner,'discover_sources',lambda *a:current['source_sha256'])
    monkeypatch.setattr(runner,'hardware',lambda:dict(memory_available_bytes=2**40,artifact_free_bytes=2**40))
    monkeypatch.setattr(runner,'create_output',lambda p:p.mkdir())
    checks=[]
    def verify(definition):
        for key in ('input_sha256','native_sha256','native_geometry_sha256','opencv_binary_sha256',
                'opencv_version','rules','native_scene_sha256'):
            assert definition[key]==launch[key]
        assert not definition['native_execution'] and not definition['model_loaded']
        assert definition['native_scene_workers']==0 and definition['memory_admission_bytes']==8*1024**3
        checks.append(True)
    monkeypatch.setattr(runner,'verify',verify)
    monkeypatch.setattr(runner,'summarize',lambda root,*a:dict(execution=dict(
        fallback_attempts=[] if root==runner.LEARNED else [dict(tick=463)])))
    runner.main()
    assert len(checks)==2 and (runner.OUTPUT/'result.json').is_file()
