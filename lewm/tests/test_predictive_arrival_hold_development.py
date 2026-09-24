from copy import deepcopy
import numpy as np
import pytest
from lewm.predictive_arrival_hold_development import select_predicted_arrival_hold
from lewm.tests.test_terminal_position_priority_development import selection


def inputs():
    result = selection() | dict(waypoint_body_xy_m=[.058, .006])
    prediction = np.zeros((6, 8, 3))
    prediction[0, -2:, :2] = [[.070, .008], [.071, .008]]
    return result, prediction


def test_predicted_quiet_goal_hold_overrides_heading_without_mutating_inputs():
    selected, prediction = inputs(); before = deepcopy(selected)
    result = select_predicted_arrival_hold(selected, prediction, arrival_radius_m=.02)
    assert result['action'] == 'hold' and result['requested_command'] == [0., 0., 0.]
    assert result['predictive_arrival_hold']['changed']
    assert not result['predictive_arrival_hold']['measured_arrival_declared']
    assert selected == before


@pytest.mark.parametrize('fault', ['outside', 'moving', 'blocked', 'recovery', 'nonfinite'])
def test_hold_requires_clear_quiet_forecast_inside_goal_and_no_active_recovery(fault):
    selected, prediction = inputs()
    if fault == 'outside': prediction[0, -1, 0] = .09
    elif fault == 'moving': prediction[0, -2:, 0] = [.060, .071]
    elif fault == 'blocked': selected['memory_forecast_candidates'][0]['nominal_predicted_path_clear'] = False
    elif fault == 'recovery': selected['clearance_turn'] = dict(active=True)
    else: prediction[0, -1, 0] = np.nan
    result = select_predicted_arrival_hold(selected, prediction, arrival_radius_m=.02)
    assert result['action'] == selected['action']
