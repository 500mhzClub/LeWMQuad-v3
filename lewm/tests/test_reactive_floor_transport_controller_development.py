"""Shared sensor/map/mission semantics and explicitly non-predictive control."""
import ast
from copy import deepcopy
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from lewm import reactive_floor_transport_controller_development as module
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportController
from lewm.measured_floor_transport_development import current_measured_floor_pose
from lewm.tests import test_joint_pulse_execution_development as fixture
from lewm.tests.test_continuous_pulse_execution_development import visual
from lewm.tests.test_measured_floor_transport_development import item
from lewm.tests.test_reactive_observed_route_development import selection

MISSION = dict(goal_initial_body_xy_m=[.2, 0.], return_initial_body_xy_m=[0., 0.], require_return_after_goal=True)


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    monkeypatch.setattr(fixture, 'visual', partial(visual, origin=1_500_000_000))


def test_same_public_packets_produce_same_transport_map_and_mission_in_warmup():
    mission = MISSION | dict(goal_initial_body_xy_m=[1., 0.])
    learned = MeasuredFloorTransportController(None, None, public_mission=mission,
        navigation_ticks=100, condition='jepa', variant='full', persistent=True)
    reactive = module.ReactiveFloorTransportController(None, public_mission=mission, navigation_ticks=100)
    previous = None
    assert not hasattr(reactive, 'model') and not hasattr(reactive, 'residual')
    for frame in range(3):
        p, d, a, raw, now, image = item(frame, previous, narrow=frame>0)
        results = []
        for c in (learned, reactive):
            c.motion = SimpleNamespace(observe=lambda *args, **kwargs:deepcopy(raw))
            results.append(c.observe(p, d, None, now_ns=now, auxiliary_depth=a, auxiliary_rgb=image))
        for result in results: assert result['terminal'] is None, result['failure']
        for key in ('evidence', 'original_visual_evidence', 'memory_receipt', 'mission_receipt', 'requested_command'):
            assert results[0][key] == results[1][key], key
        assert learned.mapper.floor == reactive.mapper.floor and learned.mapper.occupied == reactive.mapper.occupied
        assert learned.memory.route == reactive.memory.route
        position, rotation, _ = current_measured_floor_pose(results[1]['evidence'], identity=(0, 0, 0), now_ns=now)
        np.testing.assert_array_equal(reactive.memory.position, position)
        np.testing.assert_array_equal(reactive.memory.rotation, rotation)
        assert not results[1]['learned_model_used'] and not results[1]['candidate_future_outcomes_evaluated']
        previous = raw
    count = reactive.memory.partition.total_returns
    failed = reactive.observe(p, d, None, now_ns=now+100_000_000, auxiliary_depth=a, auxiliary_rgb=None)
    assert failed['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and failed['requested_command'] == [0., 0., 0.]
    assert reactive.memory.partition.total_returns == count


def prepare_mission(monkeypatch):
    c = module.ReactiveFloorTransportController(None, public_mission=MISSION, navigation_ticks=100)
    monkeypatch.setattr(module, 'causal_history_tensors', lambda h,t:deepcopy(h))
    # Pose acceptance is tested above with full packets. This fixture isolates
    # the actual settling state machine and command/mission integration.
    def pose(evidence, *, identity, now_ns):
        assert now_ns == 1_500_000_000+evidence['frame']*100_000_000
        return np.array(evidence['position'], float), np.eye(3), {'frame':evidence['frame']}
    monkeypatch.setattr(module, 'current_measured_floor_pose', pose)
    return c


def step(c, frame, position):
    return c.advance({'frame_marker':frame}, {'frame':frame, 'position':position},
        now_ns=1_500_000_000+frame*100_000_000)


def test_current_settling_mission_and_map_survive_goal_transition_without_model(monkeypatch):
    c = prepare_mission(monkeypatch); objects = (c.mapper, c.memory, c.motion, c.registration, c.history)
    reference = module.MeasuredFloorTransportMission(MISSION, navigation_ticks=100)
    c.mapper.floor[(7, 8)] = {'witness':True}
    def choose(mapper, geometry, *, now_ns):
        assert mapper is c.mapper and mapper.floor[(7, 8)] == {'witness':True}
        return selection('left_turn' if c.mission.phase == 'RETURN' else 'forward')
    monkeypatch.setattr(c.selector, 'choose', choose)
    previous = [0., 0., 0.]; first_return = None
    for frame in range(45):
        position = [0., 0., 0.] if frame < 5 or frame >= 26 else [.2, 0., 0.]
        expected = reference.advance(np.array(position), frame=frame,
            now_ns=1_500_000_000+frame*100_000_000, previous_requested_command=previous)
        result = step(c, frame, position); previous = result['requested_command']
        assert result['mission_receipt'] == expected
        if expected['phase'] == 'RETURN' and first_return is None: first_return = frame
        if result['terminal'] is not None: break
    assert first_return is not None and result['terminal'] == 'OBSERVED_ROUND_TRIP_CANDIDATE'
    assert len(result['mission_receipt']['arrivals']) == 2
    assert all(a is b for a,b in zip(objects, (c.mapper,c.memory,c.motion,c.registration,c.history), strict=True))
    assert not result['verified_round_trip'] and not result['learned_model_used']


def test_current_geometry_wait_budget_and_terminal_latch(monkeypatch):
    c = prepare_mission(monkeypatch)
    monkeypatch.setattr(c.selector, 'choose', lambda *a, **k:selection(None))
    for frame in range(13): assert step(c, frame, [0.,0.,0.])['terminal'] is None
    result = step(c, 13, [0.,0.,0.])
    assert result['terminal'] == 'NO_REACTIVE_ACTION_SATISFYING_CURRENT_GEOMETRY'
    before = deepcopy(list(c.history))
    monkeypatch.setattr(c.motion, 'observe', lambda *a, **k:pytest.fail('sensor reused after terminal'))
    result = c.observe({}, {}, {}, now_ns=2_900_000_000)
    assert result['requested_command'] == [0.,0.,0.] and list(c.history) == before


def test_controller_advance_only_changes_pose_accessor_and_full_settling_position():
    def method(path, cls, name):
        tree = ast.parse(Path(path).read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        return next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    original = method('lewm/reactive_observed_round_trip_controller_development.py', 'ReactiveObservedRoundTripController', 'advance')
    candidate = method('lewm/reactive_floor_transport_controller_development.py', 'ReactiveFloorTransportController', 'advance')
    class Normalize(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == 'current_measured_floor_pose': node.id = 'current_joint_pose'
            return node
        def visit_Call(self, node):
            if ast.unparse(node.func) == 'self.mission.advance': node.args[0] = ast.parse('p[:2]', mode='eval').body
            return self.generic_visit(node)
    assert ast.dump(original) == ast.dump(Normalize().visit(candidate))
