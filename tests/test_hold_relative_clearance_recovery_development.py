from copy import deepcopy
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.hold_relative_clearance_recovery_development import recover_against_hold


def blocked():
    return dict(action='hold', memory_forecast_candidates=[dict(action=action,
        nominal_predicted_path_clear=action == 'hold', reserve_recovery_path_clear=False,
        required_path_clearance_m=.45 if action == 'hold' else .48,
        segment_clearances_m=[.47]*8) for action in ACTIONS])


def test_gradual_recovery_beats_hold_without_requiring_full_reserve():
    original = blocked(); index = ACTIONS.index('left_arc')
    original['memory_forecast_candidates'][index]['segment_clearances_m'] = [.47]*3+[.471]*4+[.474]
    before = deepcopy(original)
    result = recover_against_hold(original)
    assert result['action'] == 'left_arc'
    assert result['selected_reserve_recovery']
    assert not result['recovery_restores_reserve_by_commit_end']
    assert original == before


def test_recovery_rejects_worse_minimum_even_with_better_endpoint():
    original = blocked()
    original['memory_forecast_candidates'][1]['segment_clearances_m'] = [.47]*3+[.469]*4+[.49]
    assert recover_against_hold(original) is original


def test_unknown_path_and_nominal_encroachment_do_not_grant_recovery():
    for path in ([None]*7+[.49], [.45]*7+[.49], [float('nan')]*7+[.49]):
        original = blocked(); original['memory_forecast_candidates'][1]['segment_clearances_m'] = path
        assert recover_against_hold(original) is original


def test_existing_clear_motion_and_selected_hold_are_preserved():
    original = blocked()
    original['memory_forecast_candidates'][1].update(nominal_predicted_path_clear=True,
        segment_clearances_m=[.49]*8)
    assert recover_against_hold(original) is original


def test_existing_action_and_insufficient_gain_are_preserved():
    original = blocked()
    original['memory_forecast_candidates'][1]['segment_clearances_m'] = [.47]*7+[.4705]
    assert recover_against_hold(original) is original
    original['action'] = 'left_turn'
    assert recover_against_hold(original) is original
