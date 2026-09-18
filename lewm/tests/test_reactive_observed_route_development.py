from types import SimpleNamespace
from functools import partial
from copy import deepcopy
import math
import numpy as np
import pytest
from lewm.reactive_observed_route_selection_development import ReactiveObservedRouteSelector
from lewm import reactive_observed_round_trip_controller_development as module
from lewm.geometry_progress_pilot_development import candidate_commands
from lewm.tests import test_joint_pulse_execution_development as joint_fixture
from lewm.tests.test_continuous_pulse_execution_development import visual


def mapper_fixture(*, heading=0., route=True, conflict=False):
    calls = []
    R = np.array([[math.cos(heading), -math.sin(heading), 0.], [math.sin(heading), math.cos(heading), 0.], [0., 0., 1.]])
    def footprint(geometry, xy, yaw, **kwargs):
        calls.append((xy, yaw, kwargs))
        return dict(possible_intersection=conflict)
    m = SimpleNamespace(map_from_initial=np.eye(3), occupied=set(),
        floor={(x, y) for x in range(-2, 20) for y in range(-2, 3)},
        surface=SimpleNamespace(position=np.array([.025, .025, 0.]), rotation=R, footprint=footprint))
    m.waypoint = lambda goal, **kwargs: dict(status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER',
        route_cells=[[x, 0] for x in range(9)] if route else [])
    return m, calls


@pytest.mark.parametrize('heading,action', [(0., 'forward'), (-math.pi/2, 'left_turn'), (math.pi/2, 'right_turn')])
def test_action_uses_observed_heading_and_current_geometry_without_model_or_future_poses(heading, action):
    m, calls = mapper_fixture(heading=heading)
    s = ReactiveObservedRouteSelector(goal_initial_body_xy_m=[1., 0.]).choose(m, object(), now_ns=1)
    assert s['action'] == action and s['requested_command'] == candidate_commands(action)[0]
    assert calls == [([0., 0.], 0., dict(now_ns=1, persistent=True))]
    assert 'prediction' not in s and 'candidates' not in s
    assert not s['learned_model_used'] and not s['candidate_future_outcomes_evaluated']


@pytest.mark.parametrize('cause', ['unknown_connector', 'occupied_current', 'occupied_connector', 'current_surface'])
def test_measured_geometry_can_reject_translation_without_predicted_candidate_filters(cause):
    m, _ = mapper_fixture(conflict=cause == 'current_surface')
    if cause == 'unknown_connector': m.floor.remove((4, 0))
    elif cause == 'occupied_current': m.occupied.add((8, 0))
    elif cause == 'occupied_connector': m.occupied.add((12, 0))
    s = ReactiveObservedRouteSelector(goal_initial_body_xy_m=[1., 0.]).choose(m, object(), now_ns=1)
    assert s['action'] is None and s['requested_command'] == [0., 0., 0.]
    assert s['current_geometry_checked'] and not s['predictive_surface_or_path_gates_applied']


def test_scan_is_bounded_and_target_transition_resets_only_scan_state():
    m, _ = mapper_fixture(route=False); selector = ReactiveObservedRouteSelector(goal_initial_body_xy_m=[1., 0.])
    floor = m.floor
    r = selector.choose(m, object(), now_ns=1)
    assert r['action'] == 'left_turn'
    for tick in range(2, 7):
        angle = selector.scan_target
        m.surface.rotation = np.array([[math.cos(angle), -math.sin(angle), 0.], [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
        r = selector.choose(m, object(), now_ns=tick)
    assert r['view_budget_exhausted'] and r['action'] is None
    selector.set_goal([0., 0.])
    assert selector.scan_index == 0 and selector.scan_target is None and m.floor is floor


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.setattr(joint_fixture, 'visual', partial(visual, origin=1_500_000_000))
    monkeypatch.setattr(module, 'causal_history_tensors', lambda h, t: deepcopy(h))
    return module.ReactiveObservedRoundTripController(object(), public_mission=dict(
        goal_initial_body_xy_m=[.2, 0.], return_initial_body_xy_m=[0., 0.], require_return_after_goal=True), navigation_ticks=40)


def selection(action):
    return dict(action=action, requested_command=[0., 0., 0.] if action is None else candidate_commands(action)[0],
        view_budget_exhausted=False, current_geometry_checked=True)


def step(c, frame, x, prior=None):
    evidence, now = joint_fixture.joint_visual(frame, (x, 0., 0.), previous=prior)
    return c.advance(dict(frame_marker=frame), evidence, now_ns=now), evidence


def test_actual_pose_mission_and_persistent_state_survive_outbound_return_without_model(controller, monkeypatch):
    c = controller; objects = (c.mapper, c.memory, c.motion, c.history)
    assert not hasattr(c, 'model') and not hasattr(c, 'residual')
    c.mapper.floor[(7, 8)] = {'witness': True}
    def choose(mapper, geometry, *, now_ns):
        assert mapper is c.mapper and (7, 8) in mapper.floor
        return selection('left_turn' if c.mission.phase == 'RETURN' else 'forward')
    monkeypatch.setattr(c.selector, 'choose', choose)
    prior = None
    for i in range(5): r, prior = step(c, i, 0., prior)
    for i in range(5, 16): r, prior = step(c, i, .2, prior)
    assert r['mission_receipt']['phase_transition'] == 'OUTBOUND_TO_RETURN'
    assert r['requested_command'] == [0., 0., 0.] and c.selector.goal.tolist() == [0., 0.]
    assert all(a is b for a, b in zip(objects, (c.mapper, c.memory, c.motion, c.history), strict=True))
    r, prior = step(c, 16, .2, prior)
    assert r['requested_command'] == [0., 0., .45] and [x['frame_marker'] for x in c.history] == [13, 14, 15, 16]
    for i in range(17, 28): r, prior = step(c, i, 0., prior)
    assert r['terminal'] == 'OBSERVED_ROUND_TRIP_CANDIDATE'
    assert [a['frame'] for a in r['mission_receipt']['arrivals']] == [15, 27]
    assert not r['verified_round_trip'] and not r['learned_model_used']


def test_ten_geometry_waits_then_stop_is_latched(controller, monkeypatch):
    c = controller; monkeypatch.setattr(c.selector, 'choose', lambda *a, **k: selection(None))
    prior = None
    for i in range(13):
        r, prior = step(c, i, 0., prior)
        assert r['terminal'] is None
    r, prior = step(c, 13, 0., prior)
    assert r['terminal'] == 'NO_REACTIVE_ACTION_SATISFYING_CURRENT_GEOMETRY'
    old_history = list(c.history)
    monkeypatch.setattr(c.selector, 'choose', lambda *a, **k: pytest.fail('resumed after terminal'))
    r, _ = step(c, 14, 0., prior)
    assert r['requested_command'] == [0., 0., 0.] and list(c.history) == old_history


def test_invalid_history_stops_before_selection_and_missing_sensor_stops_before_mission(controller, monkeypatch):
    c = controller; prior = None
    for i in range(3): r, prior = step(c, i, 0., prior)
    def fail(*a): raise ValueError('bad synthetic history')
    monkeypatch.setattr(module, 'causal_history_tensors', fail)
    monkeypatch.setattr(c.selector, 'choose', lambda *a, **k: pytest.fail('selection before history admission'))
    r, _ = step(c, 3, 0., prior)
    assert r['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and c.mission.frame == 2
    fresh = module.ReactiveObservedRoundTripController(object(), public_mission=dict(
        goal_initial_body_xy_m=[.2, 0.], return_initial_body_xy_m=[0., 0.], require_return_after_goal=True), navigation_ticks=40)
    r = fresh.observe({}, {}, {}, now_ns=1)
    assert r['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and fresh.mission.frame == -1
