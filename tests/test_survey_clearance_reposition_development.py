from copy import deepcopy

from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.survey_clearance_reposition_development import reposition_survey


def selection():
    rows = []
    for action in ACTIONS:
        path = [.49]*8 if action == 'hold' else [.47]*8
        if action == 'forward': path = [.49]*3+[.50, .51, .52, .53, .54]
        rows.append(dict(action=action, segment_clearances_m=path,
            required_path_clearance_m=.45 if action == 'hold' else .48,
            full_reserve_path_clear=action in ('hold', 'forward'),
            reserve_recovery_path_clear=False))
    return dict(action='right_turn', action_index=ACTIONS.index('right_turn'),
        requested_command=[0., 0., -.45], before_memory_filter_action='left_turn',
        scan_utilities=[], scan_heading_error_rad=1.,
        clearance_turn=dict(active=True, direction=-1, target_heading_rad=1.),
        memory_forecast_candidates=rows)


def test_reposition_preserves_target_and_forecast_evidence():
    original = selection(); before = deepcopy(original)
    result = reposition_survey(original)
    assert original == before
    assert result['action'] == 'forward'
    assert result['memory_forecast_candidates'] == original['memory_forecast_candidates']
    assert result['scan_heading_error_rad'] == original['scan_heading_error_rad']
    assert result['clearance_turn']['active'] is False
    assert result['survey_clearance_reposition']['selected_endpoint_m'] == .54


def test_no_reposition_without_full_reserve_or_improved_hold_clearance():
    for path, full in (([.479]*8, False), ([.489]*7+[.54], True),
                       ([.49]*8, True), ([None]*8, True), ([float('nan')]*8, True)):
        original = selection()
        row = next(r for r in original['memory_forecast_candidates'] if r['action']=='forward')
        row.update(segment_clearances_m=path, full_reserve_path_clear=full)
        assert reposition_survey(original) is original


def test_route_motion_and_unlatched_or_preferred_turns_are_unchanged():
    for change in ('route', 'unlatched', 'preferred'):
        original = selection()
        if change == 'route': del original['scan_utilities']
        elif change == 'unlatched': original['clearance_turn']['active'] = False
        else: original['action'] = 'left_turn'
        assert reposition_survey(original) is original
