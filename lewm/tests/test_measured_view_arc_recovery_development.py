from copy import deepcopy

import numpy as np
import pytest

from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.measured_view_arc_recovery_development import recover_view_with_arc


def stalled_view():
    prediction = np.zeros((6, 8, 5)); prediction[:, :, 3] = 1.
    i = ACTIONS.index('left_arc')
    prediction[i, 3:, 0] = np.linspace(.01, .06, 5)
    prediction[i, 3:, 2] = np.sin(np.linspace(.04, .20, 5))
    prediction[i, 3:, 3] = np.cos(np.linspace(.04, .20, 5))
    selection = dict(action='hold', scan_utilities=[], scan_heading_error_rad=.8,
        memory_forecast_candidates=[dict(action=a,
            nominal_predicted_path_clear=a in ('hold', 'left_arc', 'right_arc'),
            nominal_footprint_path_clear=True,
            reserve_recovery_path_clear=a in ('left_arc', 'right_arc'),
            clearance_check_mode='RESERVE_RECOVERY' if 'arc' in a else 'BLOCKED') for a in ACTIONS],
        planned_stopping_projection=dict(candidates=[dict(action=a, projection_clear=True) for a in ACTIONS]))
    return selection, prediction, np.zeros(3)


def test_view_stall_selects_clear_improving_arc_without_mutating_original():
    selected, prediction, reference = stalled_view()
    original = deepcopy(selected)
    result = recover_view_with_arc(selected, prediction, reference)
    assert selected == original and result['action'] == 'left_arc'
    assert result['selected_reserve_recovery']
    assert result['measured_view_arc_recovery']['stopping_projection_clear']
    assert result['memory_forecast_candidates'] == selected['memory_forecast_candidates']


@pytest.mark.parametrize('blocker', ['turn_available', 'hold_unsafe', 'arc_unsafe',
    'stopping_blocked', 'reference_far', 'forecast_leaves_reference', 'wrong_direction', 'no_gain'])
def test_no_arc_when_existing_limits_or_view_progress_disallow_it(blocker):
    selected, prediction, reference = stalled_view()
    rows = {r['action']: r for r in selected['memory_forecast_candidates']}
    if blocker == 'turn_available': rows['left_turn']['nominal_predicted_path_clear'] = True
    if blocker == 'hold_unsafe': rows['hold']['nominal_footprint_path_clear'] = False
    if blocker == 'arc_unsafe': rows['left_arc']['nominal_predicted_path_clear'] = False
    if blocker == 'stopping_blocked':
        selected['planned_stopping_projection']['candidates'][ACTIONS.index('left_arc')]['projection_clear'] = False
    if blocker == 'reference_far': reference[0] = .21
    if blocker == 'forecast_leaves_reference': prediction[ACTIONS.index('left_arc'), -1, 0] = .21
    if blocker == 'wrong_direction': selected['scan_heading_error_rad'] = -.8
    if blocker == 'no_gain':
        prediction[:, :, 2] = 0.; prediction[:, :, 3] = 1.
    assert recover_view_with_arc(selected, prediction, reference) is selected
