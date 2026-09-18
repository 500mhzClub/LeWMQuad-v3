"""Keep negative results, require the real model/audit and preserve success gates."""
from copy import deepcopy
import pytest
from scripts import run_go2_measured_plane_maze02_pilot_v1 as native


def fixture(early=False):
    audit=dict(raw_sensor_reconstruction_pass=True,raw_command_audit_pass=True,
        raw_model_command_replay_pass=True,model_state_unchanged=True,
        verified_round_trip=False,native_evaluation=dict(native_round_trip_candidate_pass=False),
        strict_physical_visibility_pass=True,hard_measurement_failed_frames=[],renderer_capture_audit={})
    receipt=dict(status='COMPLETE_ACTUAL_PREFIX_COMPARISON',actual_paired_execution_compared=True,
        physical_and_public_prefix_exact=True,all_preintervention_requested_commands_exact=True,
        complete_candidate_decisions_match_prospective_prefix=True,candidate_intervention_command_completed=True,
        intervention_command_changed=True,common_prefix_frames=123,first_changed_command_frame=122,
        physical_prefix_samples=6850,original_forecasts_compared=120,original_intervention_command=[0.,0.,0.],
        candidate_intervention_command=[0.,0.,-.45],boundary_command_samples_present=50,
        following_physical_outcomes_compared=False,navigation_verified=False,unexecuted_outcomes_inferred=False)
    collection=dict(navigation_ticks=4000,status='RESIDUAL_ANCHORED_CONTINUATION_MAZE_TERMINAL_AUDIT_REQUIRED',
        decisions=20 if early else 4000)
    if early: receipt=native.prefix_result(collection,{})
    record=dict(status=native.WORKER_STATUS,case=native.CASE[0],layout_index=native.CASE[1],
        variant=native.CASE[2],condition=native.CASE[3],model_name=native.CASE[4],
        model_state_sha256=native.inputs.job.MODEL_SHA,model_state_unchanged=True,
        measured_plane_constrained_estimator=True,collection=collection,prefix_comparison=receipt,
        **{k:deepcopy(audit[k]) for k in native.OUTCOME_KEYS})
    return record,audit


@pytest.mark.parametrize('early',[False,True])
def test_complete_negative_scientific_outcome_is_retained(early):
    record,audit=fixture(early)
    native.require_worker(record,audit)
    assert record['verified_round_trip'] is False


@pytest.mark.parametrize('fault',['model','head','budget','feature','raw','outcome','prefix','command','samples','overclaim'])
def test_missing_or_changed_evidence_rejected(fault):
    record,audit=fixture()
    if fault == 'model': record['model_state_sha256']='different'
    elif fault == 'head': record['condition']='jepa'
    elif fault == 'budget': record['collection']['navigation_ticks']=5000
    elif fault == 'feature': record['measured_plane_constrained_estimator']=False
    elif fault == 'raw': audit['raw_model_command_replay_pass']=False
    elif fault == 'outcome': record['verified_round_trip']=True
    elif fault == 'prefix': record['prefix_comparison']['physical_and_public_prefix_exact']=False
    elif fault == 'command': record['prefix_comparison']['candidate_intervention_command']=[0.,0.,0.]
    elif fault == 'samples': record['prefix_comparison']['physical_prefix_samples']=6900
    else: record['prefix_comparison']['navigation_verified']=True
    with pytest.raises(ValueError): native.require_worker(record,audit)


@pytest.mark.parametrize('visibility,hard_failure',[(False,False),(True,True)])
def test_physical_round_trip_alone_does_not_pass(visibility,hard_failure):
    record,audit=fixture()
    audit['native_evaluation']['native_round_trip_candidate_pass']=True
    audit['strict_physical_visibility_pass']=visibility
    audit['hard_measurement_failed_frames']=[30] if hard_failure else []
    for k in native.OUTCOME_KEYS: record[k]=deepcopy(audit[k])
    native.require_worker(record,audit)
    audit['verified_round_trip']=record['verified_round_trip']=True
    with pytest.raises(ValueError,match='physical and sensing'): native.require_worker(record,audit)


def test_resource_admission_preserves_reserve(monkeypatch):
    hw=dict(memory_available_bytes=32*1024**3,artifact_free_bytes=55*1024**3)
    monkeypatch.setattr(native.run,'hardware',lambda:hw)
    assert native.resources()==hw
    hw['artifact_free_bytes']-=1
    with pytest.raises(ValueError): native.resources()


def test_original_compute_and_renderer_environment_required(monkeypatch):
    for key,value in native.run.ENV.items(): monkeypatch.setenv(key,value)
    monkeypatch.delenv('LIBGL_ALWAYS_SOFTWARE',raising=False)
    launch=dict(renderer_environment={'LIBGL_ALWAYS_SOFTWARE':None})
    native.require_environment(launch)
    monkeypatch.setenv('LIBGL_ALWAYS_SOFTWARE','1')
    with pytest.raises(ValueError,match='environment'): native.require_environment(launch)
    monkeypatch.delenv('LIBGL_ALWAYS_SOFTWARE')
    monkeypatch.setenv('OMP_NUM_THREADS','8')
    with pytest.raises(ValueError,match='environment'): native.require_environment(launch)


def test_source_preflight_cannot_start_scene_or_create_attempt(monkeypatch,tmp_path):
    output=tmp_path/'attempt'
    monkeypatch.setattr(native,'OUTPUT',output)
    monkeypatch.setattr(native.run,'validate_root',lambda *a,**kw:None)
    monkeypatch.setattr(native.inputs,'prepared_sources',lambda seeds:{})
    monkeypatch.setattr(native,'resources',lambda:{'synthetic':True})
    monkeypatch.setattr(native.inputs,'admit',lambda *a:pytest.fail('full queue admission in source-only check'))
    monkeypatch.setattr(native.run,'create_output',lambda *a:pytest.fail('created attempt in source-only check'))
    monkeypatch.setattr(native,'assigned_model',lambda:pytest.fail('loaded model in source-only check'))
    native.main(source_only=True)
    assert not output.exists()
