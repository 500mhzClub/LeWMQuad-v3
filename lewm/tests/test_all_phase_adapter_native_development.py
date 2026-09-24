from copy import deepcopy
from types import SimpleNamespace
import pytest
from scripts import all_phase_adapter_native_evidence_development as evidence
from scripts.all_phase_adapter_native_startup_development import require_boundary
from scripts import run_go2_all_phase_adapter_maze02_matched_native_v1 as native


def boundary_fixture():
    decisions = [dict(requested_command=[0., 0., 0.], terminal=None, new_selection=None) for _ in range(4)]
    decisions[3] = dict(requested_command=[.16, 0., .45], terminal=None,
        new_selection=dict(prediction=[[[.01, 0., 0.]]], action='left_arc'))
    rows = [dict(tick=i, decision=d) for i, d in enumerate(decisions)]
    tape = [dict(tick=i, completed=True, requested_command=d['requested_command'],
        pre_sample_index=749+50*i, post_sample_index=799+50*i) for i, d in enumerate(decisions)]
    return rows, deepcopy(rows), tape


def test_full_forecast_decisions_and_executed_boundary_are_required():
    receipt = require_boundary(*boundary_fixture())
    assert receipt['candidate_intervention_command_completed']
    assert receipt['first_intervention_frame'] == 3


@pytest.mark.parametrize('fault', ['forecast', 'warmup_state', 'command', 'incomplete', 'terminal', 'clock', 'short', 'order'])
def test_changed_or_unexecuted_startup_is_rejected(fault):
    replay, actual, tape = boundary_fixture()
    if fault == 'forecast': actual[3]['decision']['new_selection']['prediction'][0][0][0] += .001
    if fault == 'warmup_state': actual[1]['decision']['extra_state'] = True
    if fault == 'command': tape[3]['requested_command'] = [0., 0., 0.]
    if fault == 'incomplete': tape[3]['completed'] = False
    if fault == 'terminal':
        actual[3]['decision']['terminal'] = 'SENSOR_OR_MODEL_FAILURE'
        replay[3]['decision']['terminal'] = 'SENSOR_OR_MODEL_FAILURE'
    if fault == 'clock': tape[3]['post_sample_index'] = 950
    if fault == 'short': actual.pop()
    if fault == 'order': actual[3]['tick'] = 4
    with pytest.raises(ValueError): require_boundary(replay, actual, tape)


def prefix_fixture():
    states = {c[4]: str(i) for i, c in enumerate(native.CASES)}
    reports = []
    for c in native.CASES:
        reports.append(dict(case=c[0], model_name=c[4], model_state_sha256=states[c[4]],
            frames=4, first_terminal_difference=3, model_state_unchanged=True,
            complete_original_warmup_decisions_exact=True, complete_original_decisions_reconstructed=True,
            no_recorded_observation_after_divergence_consumed=True, observed_and_contact_state_unchanged=True,
            controller_class_and_all_planning_constraints_unchanged=True, command_executed=False,
            navigation_verified=False, boundary=dict(frame=3, old_failure=evidence.prefix.EXPECTED_FAILURE,
                original_terminal='SENSOR_OR_MODEL_FAILURE', original_requested_command=[0., 0., 0.],
                candidate_terminal=None, candidate_requested_command=[.16, 0., .45],
                all_original_expanded_forward_tensors_exact=True,
                planner_forecast_matches_original_expanded_forward=True,
                recorded_predecessor_forecasts_present=False)))
    result = dict(status='ALL_PHASE_PLANNER_ADAPTER_STARTUP_COMPLETE', models=6,
        original_frames_per_model=4, first_terminal_difference=3, reports=reports,
        all_selected_forecasts_and_complete_original_outputs_exact=True,
        raw_public_startup_used=True, model_training=False, native_execution=False)
    return result, states


def test_all_six_assigned_replayed_models_admitted():
    evidence.require_prefix(*prefix_fixture())


@pytest.mark.parametrize('fault', ['missing', 'order', 'state', 'forecast', 'changed_controller', 'post_boundary', 'native_claim'])
def test_partial_changed_or_overclaimed_prefix_rejected(fault):
    result, states = prefix_fixture()
    if fault == 'missing': result['reports'].pop()
    if fault == 'order': result['reports'].reverse()
    if fault == 'state': result['reports'][0]['model_state_sha256'] = 'other'
    if fault == 'forecast': result['reports'][0]['boundary']['planner_forecast_matches_original_expanded_forward'] = False
    if fault == 'changed_controller': result['reports'][0]['controller_class_and_all_planning_constraints_unchanged'] = False
    if fault == 'post_boundary': result['reports'][0]['no_recorded_observation_after_divergence_consumed'] = False
    if fault == 'native_claim': result['reports'][0]['command_executed'] = True
    with pytest.raises(ValueError): evidence.require_prefix(result, states)


def test_old_incompatible_model_is_rejected_at_worker_admission(monkeypatch):
    case = native.CASES[0]
    monkeypatch.setattr(native, 'load_assigned', lambda *a: (SimpleNamespace(training=False), case[3], case[2]))
    with pytest.raises(ValueError, match='assigned corrected model'):
        native.assigned_model(dict(input_admission={'correction_admission': {}},
            assigned_model_states={case[4]: 'state'}), case)
