import numpy as np

from lewm.delayed_action_planning_development import ScheduledCommand
from lewm.fresh_obstacle_dispatch_development import CurrentObstacles, dispatch_request


def observation(ns,occupied=()):
    return CurrentObstacles(2,ns,(0.,0.,0.),tuple(map(tuple,np.eye(3))),frozenset(occupied),(19200,19200))


def test_new_obstacle_vetoes_plan_without_waiting_for_routing_update():
    plan=ScheduledCommand.prepare('forward',observed_ns=0,completed_ns=150_000_000)
    clear=dispatch_request(plan,observation(200_000_000),now_ns=200_000_000)
    assert clear['requested_command']==[.2,0.,0.] and clear['clearance_certified'] is False
    blocked=dispatch_request(plan,observation(200_000_000,[(4,0)]),now_ns=200_000_000)
    assert blocked['requested_command']==[0.,0.,0.] and blocked['reason']=='CURRENT_OBSERVED_OBSTACLE_VETO'


def test_stale_missing_future_observations_and_expired_commands_request_zero():
    plan=ScheduledCommand.prepare('forward',observed_ns=0,completed_ns=150_000_000)
    for obs,now in ((None,200_000_000),(observation(0),200_000_001),
            (observation(300_000_000),200_000_000),(observation(300_000_000),300_000_000)):
        assert dispatch_request(plan,obs,now_ns=now)['requested_command']==[0.,0.,0.]
