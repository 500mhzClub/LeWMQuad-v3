"""Complete negative outcomes, model-free worker calls and input authority."""
from copy import deepcopy

import numpy as np
import pytest

from scripts import run_go2_reactive_measured_plane_maze02_v1 as native


def fixture():
    audit = dict(raw_sensor_reconstruction_pass=True, raw_command_audit_pass=True,
        raw_controller_command_replay_pass=True, high_level_world_model_used=False,
        verified_round_trip=False, native_evaluation={'native_round_trip_candidate_pass': False},
        strict_physical_visibility_pass=True, hard_measurement_failed_frames=[], renderer_capture_audit={'complete': True})
    collection = dict(decisions=3, navigation_ticks=4000,
        status='REACTIVE_FLOOR_TRANSPORT_MAZE_TERMINAL_AUDIT_REQUIRED',
        learned_model_used=False, candidate_future_outcomes_evaluated=False)
    record = dict(status=native.WORKER_STATUS, case=native.CASE[0], layout_index=native.CASE[1],
        high_level_world_model_loaded=False, measured_plane_constrained_estimator=True,
        collection=collection, prefix_comparison=native.prefix_result(collection, {}),
        **{k: deepcopy(audit[k]) for k in native.OUTCOME_KEYS})
    return record, audit


def test_early_negative_retains_raw_audit_without_intervention_claim():
    record, audit = fixture(); native.require_worker(record, audit)
    assert record['prefix_comparison']['full_raw_audit_retained']
    assert not record['prefix_comparison']['actual_paired_execution_compared']


@pytest.mark.parametrize('fault', ['sensor', 'commands', 'replay', 'model_loaded', 'model_used',
    'future_used', 'budget', 'outcome', 'early_prefix', 'status'])
def test_false_raw_scope_and_promoted_early_outcome_are_rejected(fault):
    record, audit = fixture()
    if fault == 'sensor': audit['raw_sensor_reconstruction_pass'] = False
    elif fault == 'commands': audit['raw_command_audit_pass'] = False
    elif fault == 'replay': audit['raw_controller_command_replay_pass'] = False
    elif fault == 'model_loaded': record['high_level_world_model_loaded'] = True
    elif fault == 'model_used': audit['high_level_world_model_used'] = True
    elif fault == 'future_used': record['collection']['candidate_future_outcomes_evaluated'] = True
    elif fault == 'budget': record['collection']['navigation_ticks'] = 2400
    elif fault == 'outcome': record['verified_round_trip'] = True
    elif fault == 'early_prefix': record['prefix_comparison']['actual_paired_execution_compared'] = True
    elif fault == 'status': record['status'] = 'PARTIAL'
    with pytest.raises(ValueError): native.require_worker(record, audit)


def test_new_definition_does_not_inherit_model_assignment_or_planner_adapter():
    definition = native.definition()
    assert 'model_state_sha256' not in definition
    assert not definition['high_level_world_model_loaded']
    assert not definition['planner_interface_adapter_enabled']
    assert definition['planned_case'] == list(native.CASE)
    assert definition['navigation_ticks'] == 4000


@pytest.mark.parametrize('fail', [False, True])
def test_worker_uses_model_free_collection_and_audit_and_preserves_failure(tmp_path, monkeypatch, fail):
    record, audit = fixture(); calls = []
    monkeypatch.setattr(native, 'OUTPUT', tmp_path)
    directory = tmp_path/native.CASE[0]; directory.mkdir()
    np.savez(directory/'physics_trace.npz', physics_contact=np.zeros(950, dtype=int))
    launch = dict(source_sha256={native.PROTOCOL: 'protocol'}, input_admission={'prefix_report': {}})
    monkeypatch.setattr(native.run, 'read_json', lambda root, name: deepcopy(launch))
    monkeypatch.setattr(native.run, 'verify_artifacts', lambda *args: None)
    monkeypatch.setattr(native.run, 'digest', lambda path: 'synthetic')
    monkeypatch.setattr(native, 'verify_inputs', lambda launch: None)
    monkeypatch.setattr(native, 'resources', lambda: {})
    monkeypatch.setattr(native.original, 'ArticulatedCollisionGeometry', lambda path: 'geometry')
    def forbidden(): raise AssertionError('high-level model loader must never run')
    monkeypatch.setattr(native.original, 'assigned_model', forbidden)
    monkeypatch.setattr(native.original, 'case_readout', lambda *args: {'evaluator_only': True})
    def collect(index, protocol, *, output, geometry, episode_name):
        calls.append('collect')
        assert index == 2 and protocol == 'protocol' and geometry == 'geometry'
        if fail: raise RuntimeError('synthetic physical collection failure')
        return deepcopy(record['collection'])
    def replay(index, result, protocol, *, input_root, robot_geometry, episode_name):
        calls.append('audit'); assert robot_geometry == 'geometry'
        return deepcopy(audit)
    monkeypatch.setattr(native.pipeline, 'collect', collect)
    monkeypatch.setattr(native.pipeline, 'audit', replay)
    monkeypatch.setattr(native.pipeline, 'artifacts', lambda index, result: ['result.json'])
    result = native.worker('launch')
    if fail:
        assert result['status'] == 'REACTIVE_MEASURED_PLANE_WORKER_FAILED'
        assert 'synthetic physical collection failure' in result['failure']
        assert calls == ['collect']
    else:
        assert result['status'] == native.WORKER_STATUS
        assert result['high_level_world_model_loaded'] is False
        assert calls == ['collect', 'audit']
        assert result['readout'] == {'evaluator_only': True}
    assert (tmp_path/(native.CASE[0]+'_worker_terminal.json')).is_file()
    assert result['worker_log_sha256'] == 'synthetic'


def waiter_fixture(success=0):
    launch = dict(source_sha256={'frozen': 'source'})
    report = dict(complete_native_worker_and_artifact_roster_verified=True,
        actual_physical_prefix_reconstructed=True, scientific_success_required=False,
        raw_controller_audit_reexecuted=False, nominal_measured_round_trip_successes=success)
    result = dict(status='NOMINAL_MEASURED_PLANE_NATIVE_WAIT_V1_COMPLETE',
        source_sha256=launch['source_sha256'], artifact_sha256={'launch.json': native.inputs.NOMINAL_WAIT_LAUNCH_SHA},
        automatic_retry=False, navigation_qualified=False, real_time_qualified=False,
        hardware_qualified=False, goal_achieved=False, report=report)
    return result, launch


@pytest.mark.parametrize('success', [0, 1])
def test_nominal_admission_requires_execution_completion_without_selecting_success(success):
    result, launch = waiter_fixture(success)
    assert native.inputs.require_waiter_result(result, launch) == result['report']


@pytest.mark.parametrize('fault', ['status', 'source', 'launch', 'audit', 'physical_prefix', 'success_selection', 'bool_count'])
def test_nominal_partial_or_mismatched_verification_is_rejected(fault):
    result, launch = waiter_fixture()
    if fault == 'status': result['status'] = 'PARTIAL'
    elif fault == 'source': result['source_sha256'] = {'changed': 'source'}
    elif fault == 'launch': result['artifact_sha256']['launch.json'] = 'changed'
    elif fault == 'audit': result['report']['complete_native_worker_and_artifact_roster_verified'] = False
    elif fault == 'physical_prefix': result['report']['actual_physical_prefix_reconstructed'] = False
    elif fault == 'success_selection': result['report']['scientific_success_required'] = True
    elif fault == 'bool_count': result['report']['nominal_measured_round_trip_successes'] = False
    with pytest.raises(ValueError): native.inputs.require_waiter_result(result, launch)
