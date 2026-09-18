"""Causal command boundary and same-model forecast semantics, without a model run."""
from copy import deepcopy
import pytest
from scripts import measured_plane_controller_prefix_comparison_development as check


def example():
    selection = dict(action='hold', prediction=[[[0.,0.,0.,1.,-30.]]*8]*6,
        head='direct', input_variant='no_rgb', target_offsets_ns=[100_000_000*i for i in range(1,9)],
        model_prediction_corrected=True, translation_bias_training_only=True, translation_bias_xy_m=[.001,.002])
    original = dict(controller=check.ORIGINAL, tick=3, terminal=None, failure=None,
        mission_receipt=dict(frame=3,hold_required=False),
        requested_command=[0.,0.,0.], new_selection=selection, original_visual_evidence={'pose':'original'},
        evidence={'floor':'original'})
    candidate = deepcopy(original)
    candidate.update(controller=check.CANDIDATE, measured_plane_constrained_estimator=True,
        original_visual_evidence={'pose':'candidate'}, evidence={'floor':'candidate'})
    observer = dict(original=deepcopy(original['original_visual_evidence']),
        candidate=deepcopy(candidate['original_visual_evidence']), candidate_floor=deepcopy(candidate['evidence']))
    return original, candidate, deepcopy(original), observer


def test_estimates_can_change_while_requested_command_is_identical():
    values = example()
    row = check.compare(*values, frame=3)
    assert row['original_forecast_compared'] and not row['stop']
    assert not row['changed_command_executed'] and not row['navigation_recovered']


def test_first_actual_changed_request_stops_before_its_outcome():
    values = example()
    values[1]['new_selection']['action'] = 'forward'
    values[1]['requested_command'] = [.2,0.,0.]
    row = check.compare(*values, frame=3)
    assert row['stop'] and row['requested_command_changed']
    assert row['stop_reason'] == 'FIRST_CHANGED_REQUEST_OR_TERMINAL'
    assert not row['following_unexecuted_outcome_consumed']


def test_candidate_failure_stops_even_when_zero_command_matches_original():
    values = example()
    values[1].update(terminal='SENSOR_OR_MODEL_FAILURE', failure='synthetic gate', new_selection=None)
    row = check.compare(*values, frame=3)
    assert row['terminal_changed'] and row['stop'] and not row['requested_command_changed']


def test_current_mission_settling_hold_can_omit_forecast_without_inventing_one():
    values = example()
    values[1]['new_selection'] = None
    values[1]['mission_receipt']['hold_required'] = True
    row = check.compare(*values, frame=3)
    assert not row['original_forecast_compared'] and not row['stop']
    values[1]['mission_receipt']['hold_required'] = False
    with pytest.raises(ValueError, match='mission hold'): check.compare(*values, frame=3)


@pytest.mark.parametrize('fault', ['original', 'observer', 'floor', 'command', 'prediction','bias','variant','head','clock','scope'])
def test_unexplained_changes_do_not_become_candidate_success(fault):
    original, candidate, recorded, observer = example()
    if fault == 'original': original['failure'] = 'different'
    elif fault == 'observer': observer['candidate']['pose'] = 'unproved'
    elif fault == 'floor': candidate['evidence'] = {'floor':'unproved'}
    elif fault == 'command': candidate['requested_command'] = [.2,0.,0.]
    elif fault == 'prediction': candidate['new_selection']['prediction'] = []
    elif fault == 'bias': candidate['new_selection']['translation_bias_xy_m'] = [0.,0.]
    elif fault == 'variant': candidate['new_selection']['input_variant'] = 'full'
    elif fault == 'head': candidate['new_selection']['head'] = 'jepa'
    elif fault == 'clock': candidate['tick'] = 4
    elif fault == 'scope': candidate['measured_plane_constrained_estimator'] = False
    with pytest.raises(ValueError): check.compare(original, candidate, recorded, observer, frame=3)
