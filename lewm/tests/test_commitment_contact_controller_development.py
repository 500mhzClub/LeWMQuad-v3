from copy import deepcopy

import numpy as np
import pytest

from lewm.commitment_contact_controller_development import (
    CommitmentContactController, score_commitment_contact)
from lewm.executed_waypoint_score_development import score_waypoint_execution
from lewm.tests.test_executed_waypoint_score_development import fixture
from lewm.causal_residual_final_goal_development import correct_final_goal_score
from lewm.executed_horizon_final_goal_development import score_final_goal
from lewm.tests.test_executed_horizon_final_goal_development import final_selection
from lewm.tests.test_causal_residual_final_goal_development import receipt


def scored(kind='waypoint', first_contact_high=False):
    if kind == 'waypoint':
        selection, residual = fixture()
    else:
        selection, residual = final_selection(), receipt()
    prediction = np.asarray(selection['prediction'])
    prediction[1, -1, 4] = 0.
    if first_contact_high:
        prediction[1, :, 4] = 0.
    selection['prediction'] = prediction.tolist()
    if kind == 'waypoint':
        return score_waypoint_execution(selection, residual)
    return correct_final_goal_score(score_final_goal(selection), residual)


@pytest.mark.parametrize('kind', ['waypoint', 'final_goal'])
def test_change_contact_horizon_preserves_forecasts_geometry_and_causal_progress(kind):
    original = scored(kind); before = deepcopy(original)
    result = score_commitment_contact(original)
    assert original['action'] != 'forward' and result['action'] == 'forward'
    assert original == before
    for key in ('prediction', 'surface_checks', 'nominal_action_checks', 'nominal_path_checks',
            'phase_allowed_actions', 'causal_score_residual_receipt', 'scored_pose_horizon_ns',
            'path_constraint_horizon_ns'):
        assert result[key] == original[key]
    assert result['scored_contact_horizon_ns'] == 100_000_000
    assert score_commitment_contact(result) is result


@pytest.mark.parametrize('veto', ['later_path', 'surface', 'phase', 'first_contact', 'all_paths'])
def test_original_geometric_vetoes_and_first_interval_contact_still_control_actions(veto):
    original = scored(first_contact_high=veto == 'first_contact')
    if veto == 'later_path':
        original['nominal_path_checks'][1]['all_predicted_segments_nominally_clear'] = False
    elif veto == 'surface':
        original['surface_checks'][1]['possible_intersection'] = True
    elif veto == 'phase':
        original['phase_allowed_actions'] = ['hold', 'left_turn', 'right_turn']
    elif veto == 'all_paths':
        for check in original['nominal_path_checks']:
            check['all_predicted_segments_nominally_clear'] = False
    result = score_commitment_contact(original)
    assert result['action'] != 'forward'
    if veto == 'all_paths':
        assert result['action'] is None and result['requested_command'] == [0., 0., 0.]


def test_fresh_controller_composition_and_nonintervention_paths(monkeypatch):
    from lewm.extended_return_budget_comparator_controllers_development import ExtendedReturnBudgetForecastSelector
    from lewm.stop_conditioned_settling_development import StopConditionedSettlingMission
    c = CommitmentContactController(None, None, condition='jepa', variant='full', persistent=True,
        navigation_ticks=8000, public_mission=dict(goal_initial_body_xy_m=[.2, 0.],
            return_initial_body_xy_m=[0., 0.], require_return_after_goal=True))
    original = scored()
    monkeypatch.setattr(ExtendedReturnBudgetForecastSelector, 'choose', lambda *a, **k: original)
    assert c.selector.choose(None, None, None, None, now_ns=0)['action'] == 'forward'
    assert c.selector.residual is c.residual and c.memory is c.mapper.surface
    assert isinstance(c.mission, StopConditionedSettlingMission)
    for unchanged in ({}, original | {'nominal_clearance_reentry': True}):
        assert score_commitment_contact(unchanged) is unchanged
