"""Full raw audit scope and unexecuted early outcomes cannot be promoted."""
from copy import deepcopy

import pytest

from scripts import run_go2_nominal_measured_plane_maze02_v1 as native
from lewm.independent_reactive_floor_transport_study_development import OUTCOME_KEYS


def fixture():
    audit = dict(raw_sensor_reconstruction_pass=True, raw_command_audit_pass=True,
        raw_controller_command_replay_pass=True, high_level_world_model_used=False,
        raw_model_command_replay_pass=False, raw_nominal_forecast_command_replay_pass=True,
        actual_learned_model_forward_calls=0, nominal_predictive_controller=True,
        fully_nonpredictive_controller=False, model_state_unchanged=True,
        verified_round_trip=False, native_evaluation={'native_round_trip_candidate_pass': False},
        strict_physical_visibility_pass=True, hard_measurement_failed_frames=[], renderer_capture_audit={'complete': True})
    collection = dict(decisions=3, navigation_ticks=4000,
        status='RESIDUAL_ANCHORED_CONTINUATION_MAZE_TERMINAL_AUDIT_REQUIRED')
    record = dict(status=native.WORKER_STATUS, case=native.CASE[0], layout_index=native.CASE[1],
        variant=native.CASE[2], condition=native.CASE[3], model_name=native.CASE[4],
        model_state_sha256=native.inputs.job.MODEL_SHA, model_state_unchanged=True,
        measured_plane_constrained_estimator=True, collection=collection,
        prefix_comparison=native.prefix_result(collection, {}),
        **{k: deepcopy(audit[k]) for k in OUTCOME_KEYS})
    return record, audit


def test_complete_early_negative_retains_audit_without_intervention_claim():
    record, audit = fixture()
    native.require_worker(record, audit)
    assert record['prefix_comparison']['full_raw_audit_retained']
    assert not record['prefix_comparison']['actual_paired_execution_compared']


@pytest.mark.parametrize('fault', ['raw_sensor', 'controller_replay', 'learned_replay', 'model_called',
    'bool_calls', 'model_used', 'model_state', 'nominal_replay', 'nonpredictive', 'success', 'early_prefix', 'status'])
def test_invalid_raw_scope_or_promoted_outcome_is_rejected(fault):
    record, audit = fixture()
    if fault == 'raw_sensor': audit['raw_sensor_reconstruction_pass'] = False
    elif fault == 'controller_replay': audit['raw_controller_command_replay_pass'] = False
    elif fault == 'learned_replay': audit['raw_model_command_replay_pass'] = True
    elif fault == 'model_called': audit['actual_learned_model_forward_calls'] = 1
    elif fault == 'bool_calls': audit['actual_learned_model_forward_calls'] = False
    elif fault == 'model_used': audit['high_level_world_model_used'] = True
    elif fault == 'model_state': audit['model_state_unchanged'] = False
    elif fault == 'nominal_replay': audit['raw_nominal_forecast_command_replay_pass'] = False
    elif fault == 'nonpredictive': audit['fully_nonpredictive_controller'] = True
    elif fault == 'success': record['verified_round_trip'] = True
    elif fault == 'early_prefix': record['prefix_comparison']['actual_paired_execution_compared'] = True
    elif fault == 'status': record['status'] = 'PARTIAL'
    with pytest.raises(ValueError): native.require_worker(record, audit)


def test_original_worker_code_and_complete_auditors_are_retained():
    assert native._worker.__code__ is native.original.worker.__code__
    expected = dict(OUTPUT=native.OUTPUT, PROTOCOL=native.PROTOCOL, CASE=native.CASE,
        WORKER_STATUS=native.WORKER_STATUS, inputs=native.inputs, pipeline=native.pipeline,
        verify_inputs=native.verify_inputs, prefix_result=native.prefix_result,
        require_worker=native.require_worker, assigned_model=native.assigned_model)
    assert all(native._worker.__globals__[key] is value for key, value in expected.items())
    assert all(native._worker.__globals__[key] is value
        for key, value in native.original.worker.__globals__.items() if key not in expected)


def test_source_definition_exposes_nominal_planning_and_same_budget():
    definition = native.definition()
    assert definition['navigation_ticks'] == 4000
    assert definition['output_root'] == str(native.OUTPUT)
    assert definition['assigned_forecast_source'] == 'nominal_requested_twist'
    assert definition['nominal_predictive_controller']
    assert not definition['learned_model_forward_permitted']
    assert not definition['fully_nonpredictive_controller']
    assert not definition['isolated_planning_on_off_comparison']


def test_transient_idle_observation_is_retried_before_dispatch(monkeypatch):
    calls, sleeps = [], []
    def observe():
        calls.append(True)
        if len(calls) == 1: raise PermissionError('transient process error')
    monkeypatch.setattr(native, '_original_wait_for_idle', observe)
    monkeypatch.setattr(native.time, 'sleep', sleeps.append)
    native.wait_for_idle()
    assert len(calls) == 2 and sleeps == [30]
