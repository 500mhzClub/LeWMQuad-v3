from copy import deepcopy
from types import SimpleNamespace
import pytest
from lewm.tests.test_all_phase_residual_maze02_study_development import fixture as batch_fixture
from lewm.tests.test_reached_frontier_native_prefix_development import fixture as prefix_fixture
from scripts import reached_frontier_native_inputs_development as inputs
from scripts import run_go2_reached_frontier_maze03_pilot_v1 as native


def completed_batch():
    records, audits = batch_fixture(); states = {}; ids = {}
    for i, (case, record) in enumerate(zip(inputs.batch.CASES, records, strict=True)):
        states[case[4]] = str(i)
        record.update(model_state_sha256=str(i), artifact_sha256={case[0]+'/leaf': 'sha'}, readout={'case': case[0]})
        ids.update(record['artifact_sha256'])
        record['startup_comparison'].update(complete_candidate_decisions_match_prospective_prefix=True,
            candidate_intervention_command_completed=True, all_selected_forecasts_match_prospective_prefix=True,
            no_later_counterfactual_observations_used=True, first_intervention_frame=3)
    result = dict(status='ALL_PHASE_ADAPTER_MAZE02_MATCHED_NATIVE_V1_COMPLETE',
        planner_interface_adapter_enabled=True, frontier_transition_enabled=False,
        conditions=deepcopy(records), artifact_sha256=ids, **inputs.batch.complete_cohort(records, audits))
    launch = dict(planner_interface_adapter_enabled=True, frontier_transition_enabled=False, assigned_model_states=states)
    return result, launch, records, audits, [deepcopy(r['startup_comparison']) for r in records], [deepcopy(r['readout']) for r in records]


def test_completed_negative_adapter_batch_is_admitted_without_promoting_success():
    report = inputs.require_batch(*completed_batch())
    assert report['measured_round_trip_successes'] == 0 and report['scientific_success_required'] is False


@pytest.mark.parametrize('fault', ['missing_case', 'different_state', 'incomplete_command', 'forecast',
    'raw_audit', 'artifact', 'readout', 'claim', 'intervention'])
def test_partial_or_changed_adapter_batch_cannot_be_bypassed(fault):
    result, launch, records, audits, startups, readouts = completed_batch()
    if fault == 'missing_case': records.pop()
    if fault == 'different_state': launch['assigned_model_states'][inputs.batch.CASES[0][4]] = 'other'
    if fault == 'incomplete_command': startups[0]['candidate_intervention_command_completed'] = False
    if fault == 'forecast': startups[0]['all_selected_forecasts_match_prospective_prefix'] = False
    if fault == 'raw_audit': audits[0]['raw_model_command_replay_pass'] = False
    if fault == 'artifact': result['artifact_sha256'] = {}
    if fault == 'readout': readouts[0] = None
    if fault == 'claim': result['measured_round_trip_successes'] = 1
    if fault == 'intervention': result['frontier_transition_enabled'] = True
    with pytest.raises(ValueError): inputs.require_batch(result, launch, records, audits, startups, readouts)


def completed_worker():
    prefix = prefix_fixture()[0]
    report = dict(layout_index=3, reached_frontier_transition_enabled=True, raw_sensor_reconstruction_pass=True,
        raw_model_command_replay_pass=True, raw_command_audit_pass=True, model_state_unchanged=True,
        verified_round_trip=False, native_evaluation=dict(native_round_trip_candidate_pass=False),
        strict_physical_visibility_pass=True, hard_measurement_failed_frames=[], renderer_capture_audit={})
    receipt = dict(common_prefix_frames=6, first_intervention_frame=5, physical_prefix_samples=1000,
        physical_and_public_prefix_exact=True, all_preintervention_requested_commands_exact=True,
        complete_candidate_decisions_match_prospective_prefix=True, all_compared_raw_model_forecasts_exact=True,
        candidate_intervention_command_completed=True, raw_model_forecast_comparisons=3)
    record = dict(status=native.WORKER_STATUS, case=native.CASE[0], layout_index=3, model_name=native.CASE[4],
        model_state_sha256=inputs.replay.MODEL_SHA, model_state_unchanged=True,
        collection=dict(reached_frontier_transition_enabled=True), prefix_comparison=receipt,
        **{k: deepcopy(report[k]) for k in native.OUTCOME_KEYS})
    return record, report, prefix


def test_negative_scientific_frontier_outcome_keeps_full_audit_and_boundary():
    native.require_worker(*completed_worker())


@pytest.mark.parametrize('fault', ['state', 'case', 'raw', 'physics', 'command', 'forecast_count', 'false_success'])
def test_incomplete_frontier_physics_or_audit_is_rejected(fault):
    record, report, prefix = completed_worker()
    if fault == 'state': record['model_state_sha256'] = 'other'
    if fault == 'case': record['layout_index'] = 2
    if fault == 'raw': report['raw_model_command_replay_pass'] = False
    if fault == 'physics': record['prefix_comparison']['physical_prefix_samples'] = 999
    if fault == 'command': record['prefix_comparison']['candidate_intervention_command_completed'] = False
    if fault == 'forecast_count': record['prefix_comparison']['raw_model_forecast_comparisons'] = 2
    if fault == 'false_success':
        record['verified_round_trip'] = report['verified_round_trip'] = True
    with pytest.raises(ValueError): native.require_worker(record, report, prefix)


@pytest.mark.parametrize('fault', ['state', 'condition', 'variant'])
def test_worker_cannot_change_the_original_replayed_model(monkeypatch, fault):
    model = SimpleNamespace(state_dict=lambda: {})
    monkeypatch.setattr(native, 'state_digest', lambda _: 'other' if fault == 'state' else inputs.replay.MODEL_SHA)
    monkeypatch.setattr(native, 'load_assigned', lambda *a: (model,
        'direct' if fault == 'condition' else native.CASE[3], 'no_rgb' if fault == 'variant' else native.CASE[2]))
    with pytest.raises(ValueError, match='original assigned'):
        native.assigned_model({'input_admission': {'correction_admission': {}}})
