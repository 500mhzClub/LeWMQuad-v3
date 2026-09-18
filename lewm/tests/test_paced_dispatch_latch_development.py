from threading import Lock
import numpy as np

from lewm.paced_multirate_controller_development import PacedMultirateController
from lewm.delayed_action_planning_development import ScheduledCommand
from lewm.fresh_obstacle_dispatch_development import CurrentObstacles


def test_stale_veto_cannot_be_retried_late_within_same_command_window():
    runtime=object.__new__(PacedMultirateController)
    runtime.lock=Lock();runtime.faults=[];runtime.rejected_windows={}
    runtime.served_windows=set();runtime.maximum_initial_dispatch_lateness_ns=1_000_000
    runtime.plans=[ScheduledCommand.prepare('forward',observed_ns=0,completed_ns=100_000_000)]
    runtime.latest_obstacles=None
    assert runtime.request(now_ns=200_000_000)['requested_command']==[0.,0.,0.]
    runtime.latest_obstacles=CurrentObstacles(2,200_000_000,(0.,0.,0.),tuple(map(tuple,np.eye(3))),frozenset(),(19200,19200))
    result=runtime.request(now_ns=220_000_000)
    assert result['requested_command']==[0.,0.,0.] and result['reason']=='COMMAND_WINDOW_VETO_LATCHED'
    assert runtime.request(now_ns=300_000_000)['reason']=='NO_ON_TIME_PLAN'


def test_newly_published_plan_cannot_start_one_policy_tick_late():
    runtime=object.__new__(PacedMultirateController)
    runtime.lock=Lock();runtime.faults=[];runtime.rejected_windows={}
    runtime.served_windows=set();runtime.maximum_initial_dispatch_lateness_ns=0
    runtime.plans=[ScheduledCommand.prepare('forward',observed_ns=0,completed_ns=100_000_000)]
    runtime.latest_obstacles=CurrentObstacles(2,200_000_000,(0.,0.,0.),tuple(map(tuple,np.eye(3))),frozenset(),(1,1))
    result=runtime.request(now_ns=220_000_000)
    assert result['requested_command']==[0.,0.,0.] and result['reason']=='FIRST_DISPATCH_TOO_LATE'
