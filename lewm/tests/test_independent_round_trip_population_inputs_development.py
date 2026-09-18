"""Synthetic admission boundaries; no native scenes or neural inference."""
from copy import deepcopy
from dataclasses import asdict
import pytest
from scripts import independent_round_trip_population_inputs_development as mod
from lewm.tests.test_all_phase_residual_maze02_study_development import fixture as cohort_fixture


def cohort(monkeypatch):
    records, audits = cohort_fixture(); files = {}; ids = {}; states = {}
    for index, (case, record, audit) in enumerate(zip(mod.batch.CASES, records, audits, strict=True)):
        name = case[0]; state = str(index)*64
        for arm in mod.study.ARMS:
            if arm.model_name == case[4]: state = arm.model_state_sha256
        states[case[4]] = state
        record.update(collection={'test_case': name}, readout={'verified': False},
            model_state_sha256=state, worker_log_sha256='a'*64)
        bound = [name+s for s in ('_audit.json', '_startup_comparison.json', '_readout.json')]
        bound += [name+'/result.json', name+'/physics_trace.npz']
        record['artifact_sha256'] = {n:'b'*64 for n in bound}; ids.update(record['artifact_sha256'])
        ids.update({name+s:'a'*64 for s in ('_worker.log', '_worker_terminal.json', '_parent_completion.json')})
        files[name+'_worker_terminal.json'] = deepcopy(record)
        files[name+'/result.json'] = deepcopy(record['collection'])
        files[name+'_audit.json'] = deepcopy(audit)
        files[name+'_startup_comparison.json'] = deepcopy(record['startup_comparison'])
        files[name+'_readout.json'] = deepcopy(record['readout'])
        files[name+'_parent_completion.json'] = dict(case=name,
            worker_terminal_sha256=ids[name+'_worker_terminal.json'],
            verified_round_trip=False, scientific_success_required=False)
    result = dict(status='ALL_PHASE_ADAPTER_MAZE02_MATCHED_NATIVE_V1_COMPLETE',
        conditions=records, artifact_sha256=ids, planner_interface_adapter_enabled=True,
        frontier_transition_enabled=False, **mod.batch.complete_cohort(records, audits))
    launch = dict(assigned_model_states=states, input_admission={'correction_admission': {'exact': True}})
    monkeypatch.setattr(mod, 'read_json', lambda root, name: files[name])
    monkeypatch.setattr(mod.batch, 'artifacts', lambda index, collection: ['result.json', 'physics_trace.npz'])
    return result, launch, files


def test_all_six_audited_negative_results_are_admitted(monkeypatch):
    result, launch, _ = cohort(monkeypatch)
    summary = mod.require_batch(result, launch)
    assert summary['measured_round_trip_successes'] == 0
    assert summary['all_fixed_cases_executed'] is True
    assert len(summary['outcomes']) == 6
    assert all(r['hard_measurement_failed_frames'] == [7] for r in summary['outcomes'])


@pytest.mark.parametrize('fault', ['missing_case', 'reordered_case', 'wrong_status', 'wrong_model',
    'model_substitution', 'missing_physics', 'missing_worker', 'missing_parent', 'artifact_digest',
    'worker_mismatch', 'collection_mismatch', 'startup_mismatch', 'readout_mismatch', 'log_digest',
    'parent_mismatch', 'raw_audit', 'false_success', 'summary', 'frontier_policy', 'bool_completion'])
def test_changed_incomplete_or_substituted_cohort_rejected(monkeypatch, fault):
    result, launch, files = cohort(monkeypatch)
    record = result['conditions'][0]; name = record['case']
    if fault == 'missing_case': result['conditions'].pop()
    elif fault == 'reordered_case': result['conditions'].reverse()
    elif fault == 'wrong_status': result['status'] = 'INCOMPLETE'
    elif fault == 'wrong_model': record['model_state_sha256'] = 'c'*64
    elif fault == 'model_substitution':
        record['model_state_sha256'] = launch['assigned_model_states'][record['model_name']] = 'c'*64
        files[name+'_worker_terminal.json'] = deepcopy(record)
    elif fault == 'missing_physics': result['artifact_sha256'].pop(name+'/physics_trace.npz')
    elif fault == 'missing_worker': result['artifact_sha256'].pop(name+'_worker_terminal.json')
    elif fault == 'missing_parent': result['artifact_sha256'].pop(name+'_parent_completion.json')
    elif fault == 'artifact_digest': record['artifact_sha256'][name+'/result.json'] = 'c'*64
    elif fault == 'worker_mismatch': files[name+'_worker_terminal.json']['model_name'] = 'wrong'
    elif fault == 'collection_mismatch': files[name+'/result.json']['test_case'] = 'wrong'
    elif fault == 'startup_mismatch': files[name+'_startup_comparison.json']['common_prefix_frames'] = 3
    elif fault == 'readout_mismatch': files[name+'_readout.json']['verified'] = True
    elif fault == 'log_digest': record['worker_log_sha256'] = 'c'*64
    elif fault == 'parent_mismatch': files[name+'_parent_completion.json']['scientific_success_required'] = True
    elif fault == 'raw_audit': files[name+'_audit.json']['raw_model_command_replay_pass'] = False
    elif fault == 'false_success': record['verified_round_trip'] = True
    elif fault == 'summary': result['measured_round_trip_successes'] = 1
    elif fault == 'frontier_policy': result['frontier_transition_enabled'] = True
    else: result['all_fixed_cases_executed'] = 1
    with pytest.raises(ValueError): mod.require_batch(result, launch)


def test_partial_batch_is_rejected_before_expensive_or_runtime_work(monkeypatch):
    def missing(*args): raise ValueError('original result missing')
    def forbidden(*args, **kwargs): raise AssertionError('work before original completion')
    monkeypatch.setattr(mod, 'completed', missing)
    monkeypatch.setattr(mod, 'admit_static', forbidden)
    monkeypatch.setattr(mod.batch, 'verify_inputs', forbidden)
    with pytest.raises(ValueError, match='missing'): mod.admit('a'*64, {})


def test_full_admission_reexecutes_original_inputs_without_permitting_launch(monkeypatch):
    result, launch, _ = cohort(monkeypatch)
    static = dict(factory_correction_admission=launch['input_admission']['correction_admission'],
        completed_artifact_bindings=[], final_policy_review_completed=False,
        population_execution_permitted=False, new_layout_sensor_data_consumed=False)
    calls = []
    monkeypatch.setattr(mod, 'completed', lambda *args: (result, launch, {'result.json': 'd'*64}))
    monkeypatch.setattr(mod, 'admit_static', lambda *args: deepcopy(static))
    monkeypatch.setattr(mod, 'verify', lambda *args: None)
    monkeypatch.setattr(mod.batch, 'verify_inputs', lambda value, **kw: calls.append((value, kw)))
    admitted = mod.admit('d'*64, {})
    assert calls == [(launch, {'full': True})]
    assert admitted['completed_six_case_admission_performed'] is True
    assert admitted['complete_input_admission_performed'] is True
    assert admitted['all_scientific_failures_retained'] is True
    assert admitted['final_policy_review_completed'] is False
    assert admitted['population_execution_permitted'] is False
    assert admitted['development_summary']['measured_round_trip_successes'] == 0


def test_factory_and_native_correction_admissions_must_match(monkeypatch):
    result, launch, _ = cohort(monkeypatch)
    monkeypatch.setattr(mod, 'completed', lambda *args: (result, launch, {}))
    monkeypatch.setattr(mod, 'admit_static', lambda *args: {'factory_correction_admission': {'wrong': True}})
    with pytest.raises(ValueError, match='corrected-model'): mod.admit('d'*64, {})


def startup():
    reports = []; streams = []
    for case in mod.study.CASES[:4]:
        arm = mod.study.require_case(case); learned = arm.name != 'reactive'; rows = []
        for tick in range(4):
            decision = dict(terminal=None, requested_command=[0., 0., 0.], new_selection=None)
            if tick == 3: decision.update(requested_command=[.2, 0., 0.], new_selection={'action': 'forward'})
            old = deepcopy(decision)
            if learned and tick == 3: old.update(terminal='SENSOR_OR_MODEL_FAILURE', requested_command=[0., 0., 0.], new_selection=None)
            rows.append(dict(tick=tick, decision=decision, old_factory_decision=old,
                public_input_arrays_unchanged=True, complete_retained_contact_state_equal=True,
                same_assigned_study_public_mission=True, original_native_decision_reconstruction_claimed=False))
        streams.append(rows); old = rows[-1]['old_factory_decision']; new = rows[-1]['decision']
        reports.append(dict(arm=arm.name, assigned_case=case.name, original_public_packet_source=mod.factory.INPUT_CASE,
            controller_class=arm.implementation, frames=4, model_state_sha256=arm.model_state_sha256,
            model_state_unchanged=True, all_original_expanded_forward_outputs_exact=learned,
            actual_planner_forecast_exact=learned, reactive_path_unchanged=not learned,
            no_packet_after_new_planning_command_consumed=True, command_executed=False,
            study_public_mission_instantiated=True, independent_layout_sensor_data_consumed=False,
            independent_layout_navigation_execution=False, navigation_verified=False,
            boundary=dict(original_factory_terminal=old['terminal'], adapter_factory_terminal=new['terminal'],
                original_factory_command=old['requested_command'], adapter_factory_command=new['requested_command'],
                selected_action=new['new_selection']['action'])))
    result = dict(status='INDEPENDENT_ADAPTER_FACTORY_STARTUP_V1_COMPLETE', reports=reports,
        native_execution=False, model_training=False, new_layout_sensor_data_consumed=False,
        new_layout_navigation_execution=False)
    launch = dict(planned_cases=[asdict(c) for c in mod.study.CASES[:4]])
    return result, launch, streams


def test_complete_four_arm_factory_metadata_and_streams():
    mod.require_factory_reports(*startup())


@pytest.mark.parametrize('fault', ['missing_arm', 'wrong_model', 'wrong_controller', 'claimed_native',
    'extra_packet', 'warmup_command', 'reactive_difference', 'changed_arrays', 'boundary', 'forecast'])
def test_factory_scope_and_actual_saved_boundaries_cannot_drift(fault):
    r, launch, rows = startup()
    if fault == 'missing_arm': r['reports'].pop()
    elif fault == 'wrong_model': r['reports'][0]['model_state_sha256'] = 'f'*64
    elif fault == 'wrong_controller': r['reports'][0]['controller_class'] = 'Other'
    elif fault == 'claimed_native': r['new_layout_sensor_data_consumed'] = True
    elif fault == 'extra_packet': rows[0].append(rows[0][-1])
    elif fault == 'warmup_command': rows[0][0]['decision']['requested_command'] = [.2, 0., 0.]
    elif fault == 'reactive_difference': rows[2][3]['old_factory_decision']['requested_command'] = [0., 0., 0.]
    elif fault == 'changed_arrays': rows[0][0]['public_input_arrays_unchanged'] = False
    elif fault == 'boundary': r['reports'][0]['boundary']['selected_action'] = 'hold'
    else: r['reports'][0]['actual_planner_forecast_exact'] = False
    with pytest.raises(ValueError): mod.require_factory_reports(r, launch, rows)
