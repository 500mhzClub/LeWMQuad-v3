from copy import deepcopy
from types import SimpleNamespace
import pytest
from scripts import run_go2_direct_flow_commitment_contact_maze02_pilot_v1 as native
from scripts import direct_flow_commitment_contact_native_inputs_development as inputs


def fixture():
    from lewm.tests.test_direct_flow_commitment_contact_native_prefix_development import fixture as prefix_fixture
    prefix = prefix_fixture()[0]
    prefix['boundary_selected_action']='left_turn'
    prefix['boundary_requested_command']=[0.,0.,.45]
    prefix['boundary_comparison']['requested_command_changed']=True
    audit = dict(layout_index=2, ordinary_waypoint_commitment_contact_enabled=True, direct_corner_flow_missingness_fallback_enabled=True,
        raw_sensor_reconstruction_pass=True, raw_command_audit_pass=True, raw_model_command_replay_pass=True,
        model_state_unchanged=True, native_evaluation=dict(native_round_trip_candidate_pass=False),
        strict_physical_visibility_pass=True, hard_measurement_failed_frames=[], renderer_capture_audit={},
        verified_round_trip=False)
    receipt = dict(common_prefix_frames=562, first_intervention_frame=561, physical_prefix_samples=28800,
        original_forecasts_compared=558, physical_and_public_prefix_exact=True,
        all_preintervention_requested_commands_exact=True, complete_candidate_decisions_match_prospective_prefix=True,
        observer_and_full_controller_intervention=True, candidate_intervention_command_completed=True,
        intervention_command_changed=True, recovered_observation_frame=561, boundary_command_samples_present=50, navigation_verified=False,
        candidate_intervention_command=[0., 0., .45],
        following_physical_outcomes_compared=False, unexecuted_outcomes_inferred=False)
    record = dict(status=native.WORKER_STATUS, case=native.CASE[0], layout_index=2,
        model_name=native.CASE[4], condition='supervised_rollout', variant='full',
        model_state_sha256=inputs.replay.MODEL_SHA, model_state_unchanged=True,
        collection=dict(status='DIRECT_FLOW_COMMITMENT_CONTACT_MAZE02_TERMINAL_AUDIT_REQUIRED', ordinary_waypoint_commitment_contact_enabled=True, direct_corner_flow_missingness_fallback_enabled=True),
        prefix_comparison=receipt, **{k:deepcopy(audit[k]) for k in native.OUTCOME_KEYS})
    return record, audit, prefix


def test_scientific_failure_with_full_raw_and_physical_evidence_is_retained():
    native.require_worker(*fixture())


@pytest.mark.parametrize('fault', ['model', 'variant', 'condition', 'collection', 'raw', 'intervention',
    'frames', 'physics', 'forecasts', 'command', 'counterfactual', 'false_success', 'audit_controller', 'base_controller'])
def test_incomplete_changed_or_overclaimed_native_result_is_rejected(fault):
    record, audit, prefix = fixture()
    if fault == 'audit_controller': audit['direct_corner_flow_missingness_fallback_enabled'] = False
    if fault == 'base_controller': record['collection']['ordinary_waypoint_commitment_contact_enabled'] = False
    if fault == 'model': record['model_state_sha256'] = 'other'
    if fault == 'variant': record['variant'] = 'no_rgb'
    if fault == 'condition': record['condition'] = 'jepa'
    if fault == 'collection': record['collection']['direct_corner_flow_missingness_fallback_enabled'] = False
    if fault == 'raw': audit['raw_model_command_replay_pass'] = False
    if fault == 'intervention': record['prefix_comparison']['candidate_intervention_command_completed'] = False
    if fault == 'frames': record['prefix_comparison']['common_prefix_frames'] = 405
    if fault == 'physics': record['prefix_comparison']['physical_prefix_samples'] = 20999
    if fault == 'forecasts': record['prefix_comparison']['original_forecasts_compared'] = 402
    if fault == 'command': record['prefix_comparison']['candidate_intervention_command'] = [0., 0., -.45]
    if fault == 'counterfactual': record['prefix_comparison']['following_physical_outcomes_compared'] = True
    if fault == 'false_success': record['verified_round_trip'] = audit['verified_round_trip'] = True
    with pytest.raises(ValueError): native.require_worker(record, audit, prefix)


def test_incompatible_original_wrapper_rejected_before_execution(monkeypatch):
    monkeypatch.setattr(native, 'load_assigned', lambda *a: (SimpleNamespace(training=False), 'jepa', 'full'))
    with pytest.raises(ValueError, match='assigned'):
        native.assigned_model(dict(input_admission=dict(correction_admission={})))


def test_source_preflight_never_admits_inputs_or_creates_runtime_output(monkeypatch, tmp_path):
    monkeypatch.setattr(native, 'OUTPUT', tmp_path/'uncreated')
    monkeypatch.setattr(native, 'validate_root', lambda *a, **k: None)
    monkeypatch.setattr(native, 'prepared_sources', lambda: {})
    monkeypatch.setattr(native, 'hardware', lambda: dict(memory_available_bytes=64*1024**3, artifact_free_bytes=100*1024**3))
    monkeypatch.setattr(native.inputs, 'admit', lambda *a: pytest.fail('no input admission'))
    monkeypatch.setattr(native, 'create_output', lambda *a: pytest.fail('no output creation'))
    monkeypatch.setattr('sys.argv', ['native', '--source-preflight-only'])
    native.main()
    assert not native.OUTPUT.exists()


def launch_fixture(monkeypatch):
    monkeypatch.setattr(native, 'verify_ordered_launch', lambda launch: None)
    monkeypatch.setattr(native, 'digest', lambda path: 'urdf')
    admission = dict(prefix_report=fixture()[2], raw_prefix_result_sha256='raw',
        sustained_wait_result_sha256='sustained')
    return dict(planned_case=list(native.CASE), output_root=str(native.OUTPUT),
        implementation_class='DirectFlowCommitmentContactController',
        scene_specification=native.specification(2), public_mission=native.public_mission(2),
        model_state_sha256=inputs.replay.MODEL_SHA, robot_urdf_sha256='urdf',
        navigation_ticks=native.NAVIGATION_TICKS, native_scene_workers=1, opencv_threads=1, blas_threads=1,
        maximum_tasks_per_process=1, physics_paused_during_compute=True,
        renderer_capture_witnesses_enabled=True, fresh_controller_and_memory=True,
        ordinary_waypoint_commitment_contact_enabled=True, direct_corner_flow_missingness_fallback_enabled=True,
        planner_interface_adapter_enabled=True, native_execution=True, model_training=False,
        independent_layout_development_execution=False, reused_development_layout=True,
        source_sha256={'source':'sha'}, input_admission=admission)


@pytest.mark.parametrize('key,value', [
    ('implementation_class', 'HoldReorientationController'), ('navigation_ticks', 4000),
    ('model_state_sha256', 'changed'), ('native_scene_workers', 2),
    ('ordinary_waypoint_commitment_contact_enabled', False), ('direct_corner_flow_missingness_fallback_enabled', False),
    ('physics_paused_during_compute', False), ('renderer_capture_witnesses_enabled', False),
    ('fresh_controller_and_memory', False), ('model_training', True),
    ('independent_layout_development_execution', True), ('opencv_threads', True),
])
def test_changed_native_definition_rejected_before_input_access(monkeypatch, key, value):
    launch = launch_fixture(monkeypatch); launch[key] = value
    monkeypatch.setattr(inputs, 'verify_bound', lambda *a: pytest.fail('changed definition admitted'))
    with pytest.raises(ValueError, match='definition'): native.verify_inputs(launch)


def test_full_verification_reexecutes_exact_raw_and_completed_sustained_admission(monkeypatch):
    launch = launch_fixture(monkeypatch); calls = []
    monkeypatch.setattr(inputs, 'verify_bound', lambda *a: calls.append(('bound', a)))
    def admit(*args):
        calls.append(('full', args)); return deepcopy(launch['input_admission'])
    monkeypatch.setattr(inputs, 'admit', admit)
    native.verify_inputs(launch, full=True)
    assert calls == [('bound', (launch['input_admission'], launch['source_sha256'])),
        ('full', ('raw', 'sustained', launch['source_sha256']))]
    monkeypatch.setattr(inputs, 'admit', lambda *a: {'changed':True})
    with pytest.raises(ValueError, match='admission changed'): native.verify_inputs(launch, full=True)


def test_live_prerequisite_rejects_launch_before_model_or_output(monkeypatch, tmp_path):
    monkeypatch.setattr(native, 'OUTPUT', tmp_path/'uncreated')
    monkeypatch.setattr(native, 'validate_root', lambda *a, **k: None)
    monkeypatch.setattr(native, 'prepared_sources', lambda: {})
    monkeypatch.setattr(native, 'hardware', lambda: dict(memory_available_bytes=64*1024**3,
        artifact_free_bytes=100*1024**3))
    def live(*args): raise ValueError('original sustained-turn native waiter is still live')
    monkeypatch.setattr(inputs, 'admit', live)
    monkeypatch.setattr(native, 'assigned_model', lambda *a: pytest.fail('no model construction'))
    monkeypatch.setattr(native, 'create_output', lambda *a: pytest.fail('no output creation'))
    monkeypatch.setattr('sys.argv', ['native', '--raw-prefix-result-sha256', 'raw',
        '--sustained-wait-result-sha256', 'sustained'])
    with pytest.raises(ValueError, match='still live'): native.main()
    assert not native.OUTPUT.exists()


@pytest.mark.parametrize('key,value', [
    ('recovered_observation_frame',560), ('boundary_command_samples_present',49),
    ('intervention_command_changed',False), ('observer_and_full_controller_intervention',False),
    ('navigation_verified',True), ('original_forecasts_compared',558.0),
])
def test_incomplete_tracking_recovery_physics_cannot_qualify(key,value):
    record,audit,prefix=fixture();record['prefix_comparison'][key]=value
    with pytest.raises(ValueError):native.require_worker(record,audit,prefix)


def test_recovered_hold_is_not_falsely_counted_as_a_changed_command():
    record,audit,prefix=fixture()
    prefix['boundary_selected_action']=None;prefix['boundary_requested_command']=[0.,0.,0.]
    prefix['boundary_comparison']['requested_command_changed']=False
    record['prefix_comparison']['candidate_intervention_command']=[0.,0.,0.]
    record['prefix_comparison']['intervention_command_changed']=False
    native.require_worker(record,audit,prefix)
