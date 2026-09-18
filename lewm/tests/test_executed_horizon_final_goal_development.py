from copy import deepcopy
import numpy as np
import pytest
from lewm.eight_step_planning_development import plan
from lewm.tests.test_eight_step_planning_development import checked
from lewm.executed_horizon_final_goal_development import (
    score_final_goal, ExecutedHorizonFinalGoalProbe)
from lewm.exact_mission_target_goal_probe_development import ExactMissionTargetGoalProbe


def final_selection(prediction=None, cells=()):
    s = checked(prediction, cells)
    s.update(goal_body_xy_m=[.08, 0.], intermediate_target_is_mission_goal=True,
        terminal_goal_target_evidence=dict(selected=True))
    return plan(s, np.zeros(3), np.eye(3), cells)


def test_short_execution_progress_can_win_despite_full_plan_overshoot():
    p = np.asarray(final_selection()['prediction'])
    p[1, -1, 0] = .3
    s = final_selection(p); before = deepcopy(s)
    assert s['action'] == 'hold'
    r = score_final_goal(s)
    assert r['action'] == 'forward' and s == before
    assert r['scored_pose_horizon_ns'] == 100_000_000
    assert r['scored_contact_horizon_ns'] == r['path_constraint_horizon_ns'] == 800_000_000
    for key in ('prediction', 'surface_checks', 'nominal_action_checks', 'nominal_path_checks'):
        assert r[key] == s[key]
    assert r['eight_step_final_goal_candidates'] == s['candidates']


def test_terminal_contact_risk_still_vetoes_an_attractive_short_pose_score():
    p = np.asarray(final_selection()['prediction']); p[1, -1, 4] = 5.
    s = final_selection(p); r = score_final_goal(s)
    assert r['action'] == 'left_arc'
    assert r['candidates'][1]['full_plan_contact_score'] > .99
    assert r['candidates'][1]['utility_m'] < -1.


def test_later_path_surface_and_phase_vetoes_and_no_feasible_stop_survive():
    p = np.asarray(final_selection()['prediction']); p[1, 3, 0] = .2
    s = final_selection(p, [(11, 0)])
    assert s['nominal_action_checks'][1]['nominal_disk_connector_clear']
    assert not s['nominal_path_checks'][1]['all_predicted_segments_nominally_clear']
    assert score_final_goal(s)['action'] == 'left_arc'
    s['surface_checks'][2]['possible_intersection'] = True
    assert score_final_goal(s)['action'] == 'hold'
    s['phase_allowed_actions'] = ['forward']
    r = score_final_goal(s)
    assert r['action'] is None and r['requested_command'] == [0., 0., 0.]


def test_intermediate_target_and_view_selection_remain_identical():
    s = final_selection(); s['intermediate_target_is_mission_goal'] = False
    assert score_final_goal(s) is s
    s['intermediate_target_is_mission_goal'] = True; s['mode'] = 'VIEW_ACQUISITION'
    assert score_final_goal(s) is s
    exhausted = dict(view_budget_exhausted=True)
    assert score_final_goal(exhausted) is exhausted


@pytest.mark.parametrize('field,value', [
    ('actual_commitment_horizon_ns', 800_000_000),
    ('path_constraint_horizon_ns', 100_000_000),
    ('terminal_goal_target_evidence', {'selected': False}),
    ('phase_allowed_actions', [])])
def test_incompatible_horizon_or_unproved_final_target_is_rejected(field, value):
    s = final_selection(); s[field] = value
    with pytest.raises(ValueError): score_final_goal(s)


def test_arrival_budget_sensor_map_and_execution_contracts_are_inherited():
    assert ExecutedHorizonFinalGoalProbe.observe is ExactMissionTargetGoalProbe.observe
    assert ExecutedHorizonFinalGoalProbe.advance is ExactMissionTargetGoalProbe.advance
    a = ExecutedHorizonFinalGoalProbe(object(), object(), condition='jepa', variant='full', persistent=True)
    b = ExactMissionTargetGoalProbe(object(), object(), condition='jepa', variant='full', persistent=True)
    assert type(a.mapper) is type(b.mapper) and type(a.motion) is type(b.motion)
    assert type(a.memory) is type(b.memory)
