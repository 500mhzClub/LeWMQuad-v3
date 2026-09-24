import math
from threading import Lock
from types import SimpleNamespace
import numpy as np
import pytest
from lewm.signed_veto_view_recovery_development import SignedVetoViewMixin
from lewm.veto_view_round_trip_development import VetoViewRoundTripRuntime, wrapped
from lewm.stopping_margin_dispatch_development import StoppingMarginRoundTripRuntime


class Runtime(SignedVetoViewMixin, VetoViewRoundTripRuntime):
    pass


def runtime(monkeypatch, yaw, *, signed=True, reason='CURRENT_STOPPING_MARGIN_VETO'):
    cls = Runtime if signed else VetoViewRoundTripRuntime
    obj = cls.__new__(cls)
    obj.lock = Lock(); obj.view_recovery = None; obj.mission_generation = 2
    obj.plans = [SimpleNamespace(observed_ns=20, command=(.16, 0., yaw))]
    obj.heading = 0.
    def pose(*args, **kwargs):
        c, s = math.cos(obj.heading), math.sin(obj.heading)
        return None, np.array([[c,-s,0],[s,c,0],[0,0,1]]), None
    obj._pose = pose
    monkeypatch.setattr(StoppingMarginRoundTripRuntime, 'request',
        lambda self, **kwargs: dict(reason=reason, requested_command=[0.,0.,0.], command_observation_ns=20))
    monkeypatch.setattr(StoppingMarginRoundTripRuntime, '_route',
        lambda self, *args, **kwargs: dict(status='ROUTE', route_cells=[(1,2)]))
    return obj


def route(obj, now=100):
    return obj._route(SimpleNamespace(map_from_initial=np.eye(3)), None, None, measured_ns=now)


@pytest.mark.parametrize('yaw,expected', [(-.45,-math.pi/4),(.45,math.pi/4),(0.,math.pi/4)])
def test_veto_direction_and_measured_completion(monkeypatch, yaw, expected):
    obj = runtime(monkeypatch, yaw)
    request = obj.request(now_ns=100)
    assert request['reason'] == 'CURRENT_STOPPING_MARGIN_VETO'
    assert request['requested_command'] == [0.,0.,0.]
    assert request['view_recovery']['vetoed_command'] == [.16,0.,yaw]
    assert route(obj)['status'] == 'TRANSLATION_VETO_REQUIRES_NEW_VIEW'
    assert obj.scan_target == pytest.approx(expected)
    obj.heading = expected - math.copysign(.11, expected)
    assert route(obj, 101)['status'] == 'TRANSLATION_VETO_REQUIRES_NEW_VIEW'
    obj.heading = expected - math.copysign(.09, expected)
    assert route(obj, 102)['status'] == 'ROUTE'
    assert obj.view_recovery is None


def test_pretrigger_and_generation_do_not_accept_recovery(monkeypatch):
    obj = runtime(monkeypatch, -.45)
    obj.request(now_ns=100)
    assert route(obj, 99)['status'] == 'WAITING_FOR_POST_VETO_VIEW'
    assert obj.view_recovery['target_heading_rad'] is None
    obj.mission_generation += 1
    assert route(obj)['status'] == 'ROUTE'
    assert obj.view_recovery is None


def test_repeated_veto_does_not_retarget_and_angles_wrap(monkeypatch):
    obj = runtime(monkeypatch, -.45)
    obj.heading = -3.
    obj.request(now_ns=100); route(obj)
    target = wrapped(-3. - math.pi/4)
    assert obj.scan_target == pytest.approx(target)
    obj.plans[0].command = (.16,0.,.45)
    obj.request(now_ns=110); route(obj, 110)
    assert obj.scan_target == pytest.approx(target)
    assert obj.view_recovery['trigger_ns'] == 100


def test_parent_default_and_nonveto_result_unchanged(monkeypatch):
    obj = runtime(monkeypatch, -.45, signed=False)
    assert obj.request(now_ns=100)['view_recovery'] == dict(trigger_ns=100,
        mission_generation=2, target_heading_rad=None, source='actual_translation_veto')
    route(obj)
    assert obj.scan_target == pytest.approx(math.pi/4)
    obj = runtime(monkeypatch, -.45, reason='CURRENT_OBSERVATION_UNAVAILABLE_OR_STALE')
    assert obj.request(now_ns=100)['view_recovery'] is None
    assert route(obj)['status'] == 'ROUTE'
