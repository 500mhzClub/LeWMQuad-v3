import ast
import inspect
import textwrap
from copy import deepcopy
import numpy as np
import pytest
from lewm.measured_settling_round_trip_mission_development import MeasuredSettlingRoundTripMission


def mission(budget=60):
    return MeasuredSettlingRoundTripMission(dict(goal_initial_body_xy_m=[.2, 0.],
        return_initial_body_xy_m=[0., 0.], require_return_after_goal=True), navigation_ticks=budget)


def step(m, i, xyz, command=(0., 0., 0.)):
    return m.advance(xyz, frame=i, now_ns=1_500_000_000+i*100_000_000,
        previous_requested_command=command)


def test_braking_inside_goal_does_not_count_as_quiet_then_full_second_required():
    m = mission()
    for i in range(3): step(m, i, [0., 0., 0.])
    step(m, 3, [.18, 0., 0.], (.2, 0., 0.))
    r = step(m, 4, [.19, 0., 0.])
    assert r['quiet_intervals'] == 0 and not r['observed_settling']['measured_motion_quiet']
    for i in range(5, 14):
        r = step(m, i, [.19, 0., 0.]); assert not r['arrivals']
    r = step(m, 14, [.19, 0., 0.])
    assert r['phase_transition'] == 'OUTBOUND_TO_RETURN' and r['hold_required']
    assert r['arrivals'][0]['quiet_intervals'] == 10
    assert not r['observed_settling']['continuous_speed_bound'] and not r['verified_round_trip']


def test_vertical_motion_and_later_movement_reset_entire_dwell():
    m = mission()
    for i in range(3): step(m, i, [0., 0., 0.])
    for i in range(3, 8): step(m, i, [.2, 0., 0.])
    r = step(m, 8, [.2, 0., .01])
    assert r['quiet_intervals'] == 0 and not r['arrivals']
    for i in range(9, 18): assert not step(m, i, [.2, 0., .01])['arrivals']
    assert step(m, 18, [.2, 0., .01])['phase'] == 'RETURN'


def test_return_preserves_global_budget_and_requires_separate_measured_dwell():
    m = mission(budget=24)
    for i in range(3): step(m, i, [0., 0., 0.])
    for i in range(3, 14): r = step(m, i, [.2, 0., 0.])
    assert r['phase'] == 'RETURN' and len(r['arrivals']) == 1
    step(m, 14, [0., 0., 0.], (.2, 0., 0.))
    for i in range(15, 25): r = step(m, i, [0., 0., 0.])
    assert r['terminal'] == 'OBSERVED_ROUND_TRIP_CANDIDATE' and len(r['arrivals']) == 2
    assert m.navigation_ticks == 24 and not r['controller_state_reset_required']


@pytest.mark.parametrize('bad',[None, [0., 0.], [0., 0., float('nan')]])
def test_invalid_visual_position_latches_hold_without_reset(bad):
    m = mission(); step(m, 0, [0., 0., 0.])
    r = step(m, 1, bad)
    assert r['terminal'] == 'SENSOR_OR_MISSION_FAILURE' and r['hold_required']
    assert step(m, 2, [0., 0., 0.]) == r


def test_missing_frame_latches_and_input_output_are_not_aliased():
    m = mission(); p = np.zeros(3); r = step(m, 0, p)
    p[:] = 4.; r['observed_settling']['current_position_initial_body_m'][0] = 9.
    assert np.array_equal(m.previous_visual_position, np.zeros(3))
    assert m.last['observed_settling']['current_position_initial_body_m'] == [0., 0., 0.]
    assert step(m, 2, [0., 0., 0.])['terminal'] == 'SENSOR_OR_MISSION_FAILURE'


def test_budget_expires_while_motion_prevents_arrival():
    m = mission(budget=12)
    for i in range(16): r = step(m, i, [.2, 0., .01*(i%2)])
    assert r['terminal'] == 'MISSION_TICK_BUDGET_EXHAUSTED' and not r['arrivals']


def test_nonzero_request_still_resets_quiet_even_when_position_is_stationary():
    m = mission()
    for i in range(8): step(m, i, [.2, 0., 0.])
    assert step(m, 8, [.2, 0., 0.], (0., 0., .45))['quiet_intervals'] == 0


def test_controller_derivative_changes_only_mission_position_argument():
    from lewm.joint_floor_registered_controller_development import JointFloorRegisteredRoundTripController
    from lewm.measured_settling_round_trip_controller_development import MeasuredSettlingRoundTripController
    old = ast.parse(textwrap.dedent(inspect.getsource(JointFloorRegisteredRoundTripController.advance)))
    new = ast.parse(textwrap.dedent(inspect.getsource(MeasuredSettlingRoundTripController.advance)))
    replacements = 0
    for node in ast.walk(old):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'advance'
                and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'mission'):
            assert ast.unparse(node.args[0]) == 'p[:2]'
            node.args[0] = ast.Name(id='p', ctx=ast.Load()); replacements += 1
    assert replacements == 1 and ast.dump(old) == ast.dump(new)


def test_controller_uses_admitted_registered_pose_and_preserves_warmup_decision(monkeypatch):
    from types import SimpleNamespace
    from functools import partial
    from lewm.measured_settling_round_trip_controller_development import MeasuredSettlingRoundTripController
    from lewm.later_floor_resolution_controller_development import LaterFloorResolutionRoundTripController
    from lewm.tests.test_frame_floor_cache_development import equal
    from lewm.tests.test_joint_floor_registered_controller_development import packets
    from lewm.tests import test_joint_pulse_execution_development as fixture
    from lewm.tests.test_continuous_pulse_execution_development import visual
    monkeypatch.setattr(fixture, 'visual', partial(visual, origin=1_500_000_000))
    kwargs = dict(public_mission=dict(goal_initial_body_xy_m=[1.,0.],return_initial_body_xy_m=[0.,0.],
        require_return_after_goal=True), navigation_ticks=100, condition='jepa', variant='full', persistent=True)
    old, new = [cls(None, None, **kwargs) for cls in
        (LaterFloorResolutionRoundTripController, MeasuredSettlingRoundTripController)]
    previous = None
    for i in range(2):
        p, d, a, raw, now = packets(i, previous)
        old.motion = SimpleNamespace(observe=lambda *args, **kw: deepcopy(raw))
        new.motion = SimpleNamespace(observe=lambda *args, **kw: deepcopy(raw))
        x, y = [c.observe(p, d, None, now_ns=now, auxiliary_depth=a) for c in (old, new)]
        assert y['terminal'] is None, y.get('failure')
        receipt = y['mission_receipt'].pop('observed_settling')
        assert receipt['current_frame'] == i
        assert y['mission_receipt'].pop('measured_settling_required')
        assert y.pop('measured_settling_required_for_arrival')
        y['controller'] = x['controller']; equal(x, y)
        previous = raw
