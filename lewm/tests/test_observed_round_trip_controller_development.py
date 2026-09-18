from copy import deepcopy
from functools import partial
import pytest
from lewm import observed_round_trip_controller_development as module
from lewm.observed_floor_contact_development import ObservedFloorContactGoalProbe
from lewm.geometry_progress_pilot_development import candidate_commands
from lewm.tests.test_executed_horizon_final_goal_development import final_selection
from lewm.tests import test_joint_pulse_execution_development as joint_fixture
from lewm.tests.test_continuous_pulse_execution_development import visual


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.setattr(joint_fixture, 'visual', partial(visual, origin=1_500_000_000))
    # Synthetic policy payloads; pose/residual admission remains real. Separate
    # test below verifies that failure of full tensor admission latches a stop.
    monkeypatch.setattr(module, 'causal_history_tensors', lambda h, t: deepcopy(h))
    return module.ObservedRoundTripController(object(), object(),
        public_mission=dict(goal_initial_body_xy_m=[.2, 0.], return_initial_body_xy_m=[0., 0.],
            require_return_after_goal=True), navigation_ticks=40,
        condition='jepa', variant='full', persistent=True)


def forecast(action='forward'):
    s = final_selection()
    s.update(action=action, requested_command=[0., 0., 0.] if action is None else candidate_commands(action)[0],
        view_budget_exhausted=False)
    return s


def step(c, frame, x, prior=None):
    e, now = joint_fixture.joint_visual(frame, (x, 0., 0.), previous=prior)
    return c.advance(dict(frame_marker=frame), e, now_ns=now), e


def test_return_changes_goal_only_and_replans_from_retained_state(controller, monkeypatch):
    c = controller; mapper = c.mapper; memory = c.memory; motion = c.motion
    residual = c.residual; history = c.history; goals = []; histories = []
    c.mapper.floor[(7, 8)] = {'synthetic_retained_witness': True}
    def choose(model, h, m, geometry, *, now_ns):
        assert m is mapper and (7, 8) in m.floor
        goals.append(c.selector.goal.tolist()); histories.append([r['frame_marker'] for r in h])
        return forecast('left_turn' if c.mission.phase == 'RETURN' else 'forward')
    monkeypatch.setattr(c.selector, 'choose', choose)
    prior = None
    for tick in range(5): r, prior = step(c, tick, 0., prior)
    assert r['requested_command'] == [.2, 0., 0.] and r['plan_offset'] == 1
    for tick in range(5, 16): r, prior = step(c, tick, .2, prior)
    assert r['mission_receipt']['phase_transition'] == 'OUTBOUND_TO_RETURN'
    assert r['terminal'] is None and r['requested_command'] == [0., 0., 0.]
    assert r['goal_initial_body_xy_m'] == [0., 0.]
    assert c.mapper is mapper and c.memory is memory and c.motion is motion
    assert c.residual is residual and c.history is history
    r, prior = step(c, 16, .2, prior)
    assert r['requested_command'] == [0., 0., .45]
    assert goals == [[.2, 0.], [.2, 0.], [0., 0.]] and histories[-1] == [13, 14, 15, 16]
    for tick in range(17, 28): r, prior = step(c, tick, 0., prior)
    assert r['terminal'] == 'OBSERVED_ROUND_TRIP_CANDIDATE'
    assert [a['frame'] for a in r['mission_receipt']['arrivals']] == [15, 27]
    assert not r['verified_round_trip'] and c.residual.frame == 27
    before = c.residual.snapshot(); old_history = list(c.history)
    r, _ = step(c, 28, .2, prior)
    assert r['requested_command'] == [0., 0., 0.] and c.residual.snapshot() == before
    assert list(c.history) == old_history


def test_infeasible_wait_has_ten_commands_then_irrevocable_stop(controller, monkeypatch):
    c = controller
    monkeypatch.setattr(c.selector, 'choose', lambda *a, **k: forecast(None))
    prior = None
    for tick in range(13):
        r, prior = step(c, tick, 0., prior)
        assert r['terminal'] is None and r['requested_command'] == [0., 0., 0.]
        if tick >= 3: assert r['infeasible_wait_active']
    r, prior = step(c, 13, 0., prior)
    assert r['terminal'] == module.NO_FEASIBLE and r['consecutive_infeasible_observations'] == 11
    monkeypatch.setattr(c.selector, 'choose', lambda *a, **k: pytest.fail('resumed after stop'))
    r, _ = step(c, 14, 0., prior)
    assert r['terminal'] == module.NO_FEASIBLE


def test_recovery_and_global_budget_preserve_zero_stop(controller, monkeypatch):
    c = controller; c.mission.navigation_ticks = 5
    monkeypatch.setattr(c.selector, 'choose', lambda *a, **k: forecast(None if c.tick < 5 else 'forward'))
    prior = None
    for tick in range(8): r, prior = step(c, tick, 0., prior)
    assert r['feasible_action_recoveries'] == 1 and r['consecutive_infeasible_observations'] == 0
    r, prior = step(c, 8, .2, prior)
    assert r['terminal'] == 'MISSION_TICK_BUDGET_EXHAUSTED' and r['quiet_intervals'] == 0
    assert r['requested_command'] == [0., 0., 0.] and not r['mission_receipt']['arrivals']


def test_full_history_admission_failure_precedes_selection_and_latches(controller, monkeypatch):
    c = controller; prior = None
    for tick in range(3): r, prior = step(c, tick, 0., prior)
    def fail(*args): raise ValueError('synthetic invalid policy history')
    monkeypatch.setattr(module, 'causal_history_tensors', fail)
    monkeypatch.setattr(c.selector, 'choose', lambda *a, **k: pytest.fail('selected before admission'))
    r, prior = step(c, 3, 0., prior)
    assert r['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and r['requested_command'] == [0., 0., 0.]
    assert c.residual.frame == 2 and c.mission.frame == 2
    assert 'synthetic invalid policy history' in r['failure']


def test_sensor_mapping_and_visibility_observation_remain_inherited(controller):
    assert module.ObservedRoundTripController.observe is ObservedFloorContactGoalProbe.observe
    r = controller.observe({}, {}, {}, now_ns=1)
    assert r['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and r['requested_command'] == [0., 0., 0.]
    assert controller.mission.frame == -1


def test_bad_pose_cannot_transition_to_return(controller):
    c = controller
    e, now = joint_fixture.joint_visual(0)
    c.advance({}, e, now_ns=now)
    r = c.advance({}, e, now_ns=now)
    assert r['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and r['mission_receipt']['phase'] == 'OUTBOUND'
    assert not r['mission_receipt']['arrivals']
