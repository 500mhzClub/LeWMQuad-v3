from copy import deepcopy
import pytest
from lewm import all_phase_residual_maze02_study_development as mod


def fixture():
    records=[];audits=[]
    for name,index,variant,condition,model in mod.CASES:
        report=dict(layout_index=index,raw_sensor_reconstruction_pass=True,raw_command_audit_pass=True,
            raw_model_command_replay_pass=True,model_state_unchanged=True,verified_round_trip=False,
            native_evaluation={'native_round_trip_candidate_pass':False},strict_physical_visibility_pass=False,
            hard_measurement_failed_frames=[7],renderer_capture_audit={'all_frames_bound':True})
        record=dict(case=name,layout_index=index,variant=variant,condition=condition,model_name=model,
            status=mod.WORKER_STATUS,model_state_unchanged=True,
            **{k:deepcopy(report[k]) for k in mod.OUTCOME_KEYS},
            startup_comparison=dict(common_prefix_frames=4,physical_prefix_samples=900,
                physical_and_public_startup_exact=True,completed_zero_warmup_commands=3,later_physical_outcomes_compared=False))
        records.append(record);audits.append(report)
    return records,audits


def test_all_negative_scientific_results_retained():
    records,audits=fixture();summary=mod.complete_cohort(records,audits)
    assert summary['measured_round_trip_successes']==0 and len(summary['outcomes'])==6
    assert all(r['hard_measurement_failed_frames']==[7] for r in summary['outcomes'])
    assert summary['new_independent_layout_executions']==0
    assert summary['online_planning_advantage_established'] is False


@pytest.mark.parametrize('fault',['missing','reorder','seed','variant','raw_failure','false_success','partial_worker','startup'])
def test_incomplete_or_misassigned_cohort_rejected(fault):
    records,audits=fixture()
    if fault=='missing':records.pop();audits.pop()
    if fault=='reorder':records[0],records[1]=records[1],records[0]
    if fault=='seed':records[0]['model_name']='seed_2026091401_full_jepa'
    if fault=='variant':records[0]['variant']='no_rgb'
    if fault=='raw_failure':audits[0]['raw_model_command_replay_pass']=False
    if fault=='false_success':records[0]['verified_round_trip']=True
    if fault=='partial_worker':records[0]['failure']='incomplete'
    if fault=='startup':records[0]['startup_comparison']['physical_and_public_startup_exact']=False
    with pytest.raises(ValueError):mod.complete_cohort(records,audits)


def test_whole_remaining_cohort_reserve_not_single_case_allowance():
    resources=dict(memory_available_bytes=32*1024**3,artifact_free_bytes=106*1024**3)
    assert mod.resources_for(resources,6)['required_free_bytes']==106*1024**3
    resources['artifact_free_bytes']-=1
    with pytest.raises(ValueError,match='resource'):mod.resources_for(resources,6)
    assert mod.resources_for(resources,1)['required_free_bytes']==51*1024**3


@pytest.mark.parametrize('remaining',[0,7,True,1.5])
def test_remaining_size_cannot_weaken_roster(remaining):
    with pytest.raises(ValueError,match='size'):mod.resources_for({},remaining)


def test_matching_receipts_cannot_promote_failed_visibility():
    records,audits=fixture();records[0]['verified_round_trip']=audits[0]['verified_round_trip']=True
    audits[0]['native_evaluation']['native_round_trip_candidate_pass']=True
    records[0]['native_evaluation']=deepcopy(audits[0]['native_evaluation'])
    with pytest.raises(ValueError,match='joint'):mod.complete_cohort(records,audits)


def test_three_repeated_crossings_are_one_distinct_edge():
    from lewm.all_phase_residual_maze02_readout_development import traversal_counts
    forward=dict(from_cell=[-1,0],to_cell=[0,0],declared_open_edge=True)
    reverse=dict(from_cell=[0,0],to_cell=[-1,0],declared_open_edge=True)
    result=traversal_counts(dict(crossings=[forward,reverse,forward]))
    assert result['total_crossings']==3 and result['distinct_open_edges']==1 and result['invalid_crossings']==0


def test_timing_does_not_label_paused_simulation_as_realtime():
    from lewm.all_phase_residual_maze02_readout_development import timing
    result=timing([50,100,150])
    assert result['samples']==3 and result['median_ms']==100 and result['samples_above_command_interval_100ms']==1
    with pytest.raises(ValueError):timing([float('nan')])
