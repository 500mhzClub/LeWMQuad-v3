import math
from types import SimpleNamespace

import numpy as np

from lewm.initial_panorama_development import InitialSurveyMixin
from lewm.recovery_limited_initial_survey_development import RecoveryLimitedInitialSurveyMixin


class ObservedFloorRoute:
    def __init__(self):
        self.floor_snapshots = []

    def _pose(self, evidence, **kwargs):
        a = evidence.get('heading', 0.)
        rotation = np.array([[math.cos(a), -math.sin(a), 0.],
            [math.sin(a), math.cos(a), 0.], [0., 0., 1.]])
        return np.zeros(3), rotation, {}

    def _route(self, snapshot, evidence, goal, **kwargs):
        self.floor_snapshots.append(snapshot)
        return dict(status='OBSERVED_FLOOR_ROUTE', route_cells=[(0, 0)])


class Survey(InitialSurveyMixin, ObservedFloorRoute):
    pass


class OuterRecovery:
    def _route(self, snapshot, evidence, goal, **kwargs):
        result = super()._route(snapshot, evidence, goal, **kwargs)
        if evidence.get('visual_support', {}).get('recovery_state_at_observation'):
            return result | dict(status='RECOVERY_FIRST', route_cells=[])
        return result


class Limited(OuterRecovery, Survey, RecoveryLimitedInitialSurveyMixin, InitialSurveyMixin):
    pass


class Original(OuterRecovery, Survey):
    pass


def snapshot(ns):
    return SimpleNamespace(map_from_initial=np.eye(3), measured_ns=ns, frame=ns//100)


def weak(ns):
    return dict(visual_support=dict(frame=ns//100, measured_ns=ns,
        camera_cadence_recovery=True, recovery_state_at_observation=dict(trigger_ns=ns)))


def test_interrupted_sweep_stays_incomplete_and_outer_recovery_still_wins():
    runtime = Limited()
    runtime._route(snapshot(0), {}, None, measured_ns=0)
    completed = runtime.initial_panorama.state['completed_view_stages'].copy()
    current = snapshot(100)
    result = runtime._route(current, weak(100), None, measured_ns=100)
    assert result['status'] == 'RECOVERY_FIRST' and result['route_cells'] == []
    assert runtime.initial_panorama.state['completed_view_stages'] == completed
    assert not runtime.initial_panorama.complete
    assert runtime.initial_panorama.state['deferred']
    result = runtime._route(current, {}, None, measured_ns=200)
    assert result['status'] == 'OBSERVED_FLOOR_ROUTE'
    assert result['initial_survey']['complete'] is False
    assert result['initial_survey']['completed_view_stages'] == completed
    assert runtime.floor_snapshots == [current, current]


def test_uninterrupted_sweep_matches_original_at_every_stage():
    limited, original = Limited(), Original()
    for i in range(10):
        observation = dict(heading=i*math.pi/4)
        current = snapshot(i*100)
        assert limited._route(current, observation, None, measured_ns=i*100) == original._route(
            current, observation, None, measured_ns=i*100)
    assert limited.initial_panorama.complete and limited.initial_survey_deferral is None


def test_weak_first_planning_view_keeps_recovery_priority_without_inventing_survey():
    runtime = Limited()
    result = runtime._route(snapshot(0), weak(0), None, measured_ns=0)
    assert result['status'] == 'RECOVERY_FIRST'
    assert not runtime.initial_panorama.complete
    result = runtime._route(snapshot(100), weak(100), None, measured_ns=100)
    assert result['status'] == 'RECOVERY_FIRST'
    assert runtime.initial_panorama.state['deferred']
