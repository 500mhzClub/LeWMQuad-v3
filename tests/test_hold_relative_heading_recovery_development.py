from copy import deepcopy

from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.hold_relative_heading_recovery_development import (
    HoldRelativeHeadingRecoveryMixin, recover_heading_against_hold)


def selection(scan=False):
    rows = [dict(action=a, segment_clearances_m=[.4662 if a=='right_turn' else .466]*8,
        nominal_predicted_path_clear=a=='hold', reserve_recovery_path_clear=False,
        full_reserve_path_clear=a=='hold', required_path_clearance_m=.45 if a=='hold' else .48)
        for a in ACTIONS]
    candidates = [dict(action=a, utility_m=.05 if a=='right_turn' else -.01,
        predicted_heading_error_at_commit_start_rad=1.,
        predicted_heading_error_at_commit_end_rad=.8 if a=='right_turn' else 1.1)
        for a in ACTIONS]
    result = dict(action='hold', action_index=0, requested_command=[0., 0., 0.],
        before_memory_filter_action='right_turn', candidates=candidates,
        memory_forecast_candidates=rows)
    if scan:
        result.update(scan_utilities=deepcopy(candidates), scan_heading_error_rad=-1.)
    return result


def test_route_and_survey_turn_preserve_forecasts_and_small_clearance_gain():
    for scan in (False, True):
        original = selection(scan); before = deepcopy(original)
        result = recover_heading_against_hold(original)
        assert original == before and result['action'] == 'right_turn'
        assert result['requested_command'] == [0., 0., -.45]
        assert result['selected_reserve_recovery']
        assert not result['recovery_restores_reserve_by_commit_end']
        assert result['candidates'] == original['candidates']
        assert result['clearance_turn']['active'] is False
        row = result['memory_forecast_candidates'][-1]
        assert row['segment_clearances_m'] == before['memory_forecast_candidates'][-1]['segment_clearances_m']
        assert row['required_path_clearance_m'] == .48 and not row['full_reserve_path_clear']


def test_reject_clearance_loss_nonfinite_or_missing_paths():
    for path in ([.465]*8, [.467]*7+[.4659], [float('nan')]*8, [None]*8, [.467]*7):
        s = selection(); s['memory_forecast_candidates'][-1]['segment_clearances_m'] = path
        assert recover_heading_against_hold(s) is s
    s = selection(); s['memory_forecast_candidates'][0]['segment_clearances_m'] = [.45]*8
    assert recover_heading_against_hold(s) is s


def test_reject_nonpreferred_nonimproving_or_already_eligible_motion():
    for change in ('selected', 'translation', 'eligible', 'heading', 'utility'):
        s = selection()
        if change == 'selected': s['action'] = 'left_turn'
        elif change == 'translation': s['before_memory_filter_action'] = 'forward'
        elif change == 'eligible': s['memory_forecast_candidates'][1]['nominal_predicted_path_clear'] = True
        elif change == 'heading': s['candidates'][-1]['predicted_heading_error_at_commit_end_rad'] = 1.2
        else: s['candidates'][-1]['utility_m'] = -.02
        assert recover_heading_against_hold(s) is s


def test_terminal_approach_excluded_and_runtime_latch_cleared_only_on_recovery():
    class Base:
        def _select_clear_prediction(self, value): return value
    class Runtime(HoldRelativeHeadingRecoveryMixin, Base): pass
    runtime = Runtime(); runtime.clearance_turn = {'active': True}
    runtime.terminal_position_approach = True
    s = selection()
    assert runtime._select_clear_prediction(s) is s and runtime.clearance_turn
    runtime.terminal_position_approach = False
    assert runtime._select_clear_prediction(s)['action'] == 'right_turn'
    assert runtime.clearance_turn is None
