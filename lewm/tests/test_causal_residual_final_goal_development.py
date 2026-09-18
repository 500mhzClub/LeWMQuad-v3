from copy import deepcopy
import numpy as np
import pytest
from lewm.causal_residual_final_goal_development import correct_final_goal_score, CausalResidualFinalGoalProbe
from lewm.executed_horizon_final_goal_development import score_final_goal, ExecutedHorizonFinalGoalProbe
from lewm.online_executed_residual_development import OnlineExecutedResidual
from lewm.tests.test_executed_horizon_final_goal_development import final_selection
from lewm import matched_model_goal_probe_development as mission


def receipt(bias=(0., 0.)):
    s = OnlineExecutedResidual(); s.frame = 10; s.now_ns = 2_500_000_000
    s.history.append(dict(tick=9, available_tick=10, residual_xy_m=list(bias)))
    return s.snapshot()


def test_score_can_change_action_without_altering_predictions_checks_or_inputs():
    s = score_final_goal(final_selection()); before = deepcopy(s); bias = receipt((-.07, 0.))
    r = correct_final_goal_score(s, bias)
    assert s['action'] == 'forward' and r['action'] == 'hold' and s == before
    for k in ('prediction', 'surface_checks', 'nominal_action_checks', 'nominal_path_checks',
            'phase_allowed_actions', 'scored_pose_horizon_ns', 'scored_contact_horizon_ns'):
        assert r[k] == s[k]
    assert r['uncorrected_final_goal_candidates'] == s['candidates']
    assert not r['corrected_scoring_path_checked']
    bias['correction_xy_m'][0] = 999.
    assert r['causal_score_residual_receipt']['correction_xy_m'] == [-.07, 0.]


def test_surface_later_path_phase_and_contact_cost_still_control_feasibility():
    p = np.asarray(final_selection()['prediction']); p[1, 3, 0] = .2
    s = score_final_goal(final_selection(p, [(11, 0)]))
    assert correct_final_goal_score(s, receipt())['action'] == 'left_arc'
    s['surface_checks'][2]['possible_intersection'] = True
    assert correct_final_goal_score(s, receipt())['action'] == 'hold'
    s['phase_allowed_actions'] = ['forward']
    r = correct_final_goal_score(s, receipt())
    assert r['action'] is None and r['requested_command'] == [0., 0., 0.]
    p = np.asarray(final_selection()['prediction']); p[1, -1, 4] = 5.
    s = score_final_goal(final_selection(p))
    assert correct_final_goal_score(s, receipt())['candidates'][1]['utility_m'] < -1.


def test_nonfinal_scores_unchanged_and_future_or_native_receipts_rejected():
    s = final_selection(); s['intermediate_target_is_mission_goal'] = False
    assert correct_final_goal_score(s, receipt()) is s
    s = score_final_goal(final_selection())
    for key, value in (('native_outcomes_used', True), ('residual_available_ticks', [11]),
            ('residual_source_ticks', [10]), ('correction_xy_m', [float('nan'), 0.])):
        bad = receipt(); bad[key] = value
        with pytest.raises(ValueError): correct_final_goal_score(s, bad)


@pytest.mark.parametrize('arrival', [False, True])
def test_original_budget_and_arrival_dwell_end_without_resuming_residual_updates(monkeypatch, arrival):
    c = CausalResidualFinalGoalProbe(object(), object(), condition='jepa', variant='full', persistent=True)
    c.tick = 239 if arrival else 242
    if arrival: c.quiet = 9; c.was_within_goal = True
    frame = c.tick+1
    monkeypatch.setattr(c.residual, 'observe', lambda *a, **k: None)
    monkeypatch.setattr(mission, 'current_joint_pose', lambda *a, **k:
        (np.array([1.2, 0., 0.]) if arrival else np.zeros(3), np.eye(3), dict(frame=frame)))
    r = c.advance({}, None, now_ns=1_500_000_000+frame*100_000_000)
    assert r['terminal'] == ('OBSERVED_GOAL_CANDIDATE' if arrival else 'MISSION_TICK_BUDGET_EXHAUSTED')
    assert r['requested_command'] == [0., 0., 0.]
    before = c.residual.snapshot()
    monkeypatch.setattr(c.residual, 'observe', lambda *a, **k: pytest.fail('terminal state observed again'))
    assert c.advance({}, None, now_ns=1_500_000_000+(frame+1)*100_000_000)['terminal'] == r['terminal']
    assert c.residual.snapshot() == before


def test_sensor_observe_map_motion_and_failure_stop_remain_inherited():
    assert CausalResidualFinalGoalProbe.observe is ExecutedHorizonFinalGoalProbe.observe
    a = CausalResidualFinalGoalProbe(object(), object(), condition='jepa', variant='full', persistent=True)
    b = ExecutedHorizonFinalGoalProbe(object(), object(), condition='jepa', variant='full', persistent=True)
    assert type(a.mapper) is type(b.mapper) and type(a.motion) is type(b.motion)
    r = a.observe({}, {}, {}, now_ns=1)
    assert r['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and r['requested_command'] == [0., 0., 0.]
