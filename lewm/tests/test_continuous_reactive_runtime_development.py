from threading import Lock
from types import SimpleNamespace
import numpy as np
import pytest
from lewm.continuous_reactive_runtime_development import ContinuousReactiveRuntime
from lewm.clearance_preferred_reactive_runtime_development import ClearancePreferredReactiveRuntime
from lewm.paced_multirate_controller_development import PacedMultirateController
from lewm.initial_panorama_development import InitialPanorama,InitialSurveyRuntime


@pytest.mark.parametrize('runtime_type',(ContinuousReactiveRuntime,ClearancePreferredReactiveRuntime))
def test_shared_planner_schedules_reactive_view_without_history_or_forecasts(runtime_type):
    runtime=runtime_type.__new__(runtime_type)
    now=1_900_000_000
    runtime.lock=Lock();runtime.goal=np.array([1.,0.]);runtime.planning=[]
    runtime.latest_map=SimpleNamespace(measured_ns=now,frame=4,
        map_from_initial=np.eye(3),fine_occupied=frozenset())
    runtime.initial_panorama=InitialPanorama()
    runtime._pose=lambda *a,**k:(np.zeros(3),np.eye(3),None)
    runtime._prefix_commands=lambda ns:[[0.,0.,0.]]*3
    runtime.planning_delay_ticks=3;runtime.commit_ticks=4
    runtime.clock_ns=lambda:now+10_000_000
    committed=[];runtime._store_plan=lambda *args:committed.append(args)
    # No policy/history/model is supplied. The common planner must never
    # try to build a forecast when selecting this arm's initial survey turn.
    packet=SimpleNamespace(frame=4,measured_ns=now)
    PacedMultirateController._plan(runtime,(packet,{}))
    record=runtime.planning[-1]
    assert record['action']=='left_turn' and record['on_time']
    assert record['route_status']=='INITIAL_PANORAMA_REQUIRES_VIEW'
    assert record['selection']['candidate_future_outcomes_evaluated'] is False
    assert 'motion_correction' not in record
    plan=committed[0][0]
    assert plan.dispatch_ns==now+300_000_000
    assert plan.expires_ns==now+700_000_000
    assert runtime._route.__func__ is InitialSurveyRuntime._route


@pytest.mark.parametrize('runtime_type',(ContinuousReactiveRuntime,ClearancePreferredReactiveRuntime))
def test_reactive_treatment_rejects_model_and_future_filter(runtime_type):
    with pytest.raises(ValueError,match='model-free'):
        runtime_type(object(),condition='reactive')
    runtime=runtime_type.__new__(runtime_type)
    with pytest.raises(RuntimeError,match='predicted outcomes'):
        runtime._select_clear_prediction(None)


def test_route_preference_matches_learned_arm_without_constructing_a_model():
    from lewm.clearance_preferred_route_development import ClearancePreferredTurnRecoveryRuntime
    floor=frozenset((x,y) for x in range(25) for y in range(25))
    snapshot=SimpleNamespace(floor=floor,occupied=frozenset(),fine_occupied=frozenset())
    routes=[]
    for runtime_type in (ClearancePreferredReactiveRuntime,ClearancePreferredTurnRecoveryRuntime):
        runtime=runtime_type.__new__(runtime_type)
        runtime.frontier_visits=SimpleNamespace(excluded={(24,y) for y in range(25)})
        runtime.mission_latest=None
        route=runtime._routing_proposer(snapshot)(floor,frozenset(),[.5,.5],[2.,.5])
        route['clearance_preferred_route'].pop('added_routing_s')
        routes.append(route)
    assert routes[0]==routes[1]
