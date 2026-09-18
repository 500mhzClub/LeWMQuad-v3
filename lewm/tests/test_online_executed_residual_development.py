from copy import deepcopy
from functools import partial
import numpy as np
import pytest
from lewm.online_executed_residual_development import OnlineExecutedResidual
from lewm.causal_executed_residual_diagnosis_development import replay_bias
from lewm.geometry_progress_pilot_development import candidate_commands
from lewm.tests.test_executed_horizon_final_goal_development import final_selection
from lewm.tests import test_joint_pulse_execution_development as joint_fixture
from lewm.tests.test_continuous_pulse_execution_development import visual


@pytest.fixture(autouse=True)
def synthetic_mission_clock(monkeypatch):
    monkeypatch.setattr(joint_fixture, 'visual', partial(visual, origin=1_500_000_000))


def request(tick, action='forward'):
    return dict(tick=tick, terminal=None, new_selection=final_selection(),
        requested_command=candidate_commands(action)[0])


def test_online_mean_matches_fixed_prequential_diagnostic_and_owns_original_prediction():
    state = OnlineExecutedResidual(); prior = None; records = []
    for tick in range(13):
        e, now = joint_fixture.joint_visual(tick, (tick*.01, -tick*.002, 0.), previous=prior)
        state.observe(e, now_ns=now); before = state.snapshot()
        r = request(tick)
        expected = replay_bias(records+[dict(tick=tick, available_tick=tick+1,
            predicted_body_xy_m=[.05, 0.], observed_body_xy_m=[.01, -.002])])[-1]
        np.testing.assert_allclose(before['correction_xy_m'], expected['correction_xy_m'], atol=1e-15)
        assert before['residual_source_ticks'] == expected['residual_source_ticks']
        assert before['residual_available_ticks'] == expected['residual_available_ticks']
        state.remember(r)
        r['new_selection']['prediction'][1][0][0] = 99.
        assert state.pending['predicted_body_xy_m'] == [.05, 0.]
        assert state.snapshot()['correction_xy_m'] == before['correction_xy_m']
        records.append(dict(tick=tick, available_tick=tick+1,
            predicted_body_xy_m=[.05, 0.], observed_body_xy_m=[.01, -.002]))
        prior = e
    assert len(state.history) == 8


def test_requested_zero_uses_hold_forecast_even_when_selector_has_no_feasible_action():
    state = OnlineExecutedResidual(); e, now = joint_fixture.joint_visual(0)
    state.observe(e, now_ns=now); r = request(0, 'hold'); r['new_selection']['action'] = None
    state.remember(r)
    assert state.pending['action'] == 'hold' and state.pending['predicted_body_xy_m'] == [0., 0.]
    with pytest.raises(ValueError): state.remember(r)
    e1, t1 = joint_fixture.joint_visual(1, (.01, 0., .1), previous=e)
    state.observe(e1, now_ns=t1)
    np.testing.assert_allclose(state.snapshot()['correction_xy_m'], [-.01, 0.], atol=1e-15)


def test_bad_clock_and_unknown_request_cannot_add_a_residual():
    state = OnlineExecutedResidual(); e, now = joint_fixture.joint_visual(0)
    state.observe(e, now_ns=now); before = state.snapshot()
    with pytest.raises(ValueError): state.observe(e, now_ns=now)
    assert state.snapshot() == before
    r = request(0); r['requested_command'] = [.123, 0., 0.]
    with pytest.raises(ValueError): state.remember(r)
    assert state.snapshot() == before


def test_missing_forecasts_expire_without_imputing_observed_targets():
    state = OnlineExecutedResidual(); prior = None
    for tick in range(11):
        e, now = joint_fixture.joint_visual(tick, (tick*.01, 0., 0.), previous=prior)
        state.observe(e, now_ns=now)
        if tick == 0: state.remember(request(0))
        prior = e
    assert state.snapshot()['observed_residual_samples'] == 0
    assert state.snapshot()['correction_xy_m'] == [0., 0.]
