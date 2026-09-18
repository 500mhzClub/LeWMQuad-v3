from copy import deepcopy
import pytest
from lewm.hold_reorientation_prefix_comparison_development import compare_step
from lewm.tests.test_hold_reorientation_development import ready, selection, choose


def pair(frame=13, changed=True):
    s = selection(frame) if frame >= 3 else None
    old = dict(tick=frame, controller='residual_anchored_continuation_controller_v1',
        terminal=None, failure=None, requested_command=[0., 0., 0.],
        selected_action=None if s is None else s['action'], new_selection=s,
        mission_receipt=dict(arrivals=[], hold_required=False),
        evidence=dict(observed_xy=[.2, .3]), causal_residual_receipt=dict(pending_forecast_tick=frame))
    new = deepcopy(old)
    new.update(controller='hold_reorientation_controller_v1', hold_reorientation_enabled=True)
    if changed:
        new['new_selection'] = choose(ready(), frame, deepcopy(s))
        new['selected_action'] = new['new_selection']['action']
        new['requested_command'] = new['new_selection']['requested_command']
    return old, new


@pytest.mark.parametrize('frame,changed', [(0, False), (3, False), (13, True)])
def test_complete_decision_normalization_only_declared_changes(frame, changed):
    old, new = pair(frame, changed)
    before = deepcopy((old, new))
    r = compare_step(old, new, old['requested_command'], frame=frame,
        expected_selection=deepcopy(new['new_selection']))
    assert r['requested_command_changed'] is changed
    assert (old, new) == before


@pytest.mark.parametrize('field,value', [('mission_receipt', {'arrivals': ['invented']}),
    ('evidence', {'observed_xy': [1., 2.]}), ('causal_residual_receipt', {'pending_forecast_tick': 12}),
    ('selected_action', 'left_turn'), ('terminal', 'SENSOR_OR_MODEL_FAILURE'), ('failure', 'bad')])
def test_unrelated_controller_changes_rejected(field, value):
    old, new = pair(); expected = deepcopy(new['new_selection']); new[field] = value
    with pytest.raises(ValueError): compare_step(old, new, [0., 0., 0.], frame=13, expected_selection=expected)


@pytest.mark.parametrize('part', ['prediction', 'candidates', 'nominal_path_checks', 'surface_checks'])
def test_even_matching_expectation_cannot_hide_mutated_original_evidence(part):
    old, new = pair(); new['new_selection'][part] = []
    with pytest.raises(ValueError, match='complete original'):
        compare_step(old, new, [0., 0., 0.], frame=13, expected_selection=deepcopy(new['new_selection']))


def test_wrong_actual_command_or_saved_selection_expectation_rejected():
    old, new = pair()
    with pytest.raises(ValueError):
        compare_step(old, new, [0., 0., .45], frame=13, expected_selection=new['new_selection'])
    with pytest.raises(ValueError, match='frozen saved-selection'):
        compare_step(old, new, [0., 0., 0.], frame=13, expected_selection=old['new_selection'])


def test_undeclared_command_change_is_not_normalized_away():
    old, new = pair(3, False); new['requested_command'] = [0., 0., .45]
    with pytest.raises(ValueError, match='complete observed'):
        compare_step(old, new, [0., 0., 0.], frame=3, expected_selection=new['new_selection'])
