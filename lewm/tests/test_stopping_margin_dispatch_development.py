from lewm.delayed_action_planning_development import ScheduledCommand
from lewm.fresh_obstacle_dispatch_development import dispatch_request as original
from lewm.stopping_margin_dispatch_development import dispatch_request,StoppingMarginRoundTripRuntime,_StoppingDispatch
from lewm.continuous_commitment_runtime_development import ContinuousCommitmentRuntime
from lewm.paced_multirate_controller_development import PacedMultirateController
from lewm.tests.test_fresh_obstacle_dispatch_development import observation


def plan(action):
    return ScheduledCommand.prepare(action,observed_ns=0,completed_ns=200_000_000,delay_ticks=3,commit_ticks=4)


def test_stop_earlier_but_keep_original_turn_footprint_and_freshness():
    obs=observation(200_000_000,[(12,0)])
    assert original(plan('forward'),obs,now_ns=300_000_000)['requested_command']==[.2,0.,0.]
    result=dispatch_request(plan('forward'),obs,now_ns=300_000_000)
    assert result['reason']=='CURRENT_STOPPING_MARGIN_VETO' and result['requested_command']==[0.,0.,0.]
    assert result['stopping_margin_connector']['radius_m']==.45
    assert dispatch_request(plan('left_turn'),obs,now_ns=300_000_000)['requested_command']==[0.,0.,.45]
    blocked=observation(200_000_000,[(8,0)])
    assert dispatch_request(plan('left_turn'),blocked,now_ns=300_000_000)['reason']=='CURRENT_OBSERVED_OBSTACLE_VETO'
    assert dispatch_request(plan('forward'),obs,now_ns=500_000_000)['reason']=='CURRENT_OBSERVATION_UNAVAILABLE_OR_STALE'


def test_dispatch_override_keeps_commitment_history_layer():
    mro=StoppingMarginRoundTripRuntime.__mro__
    assert mro.index(ContinuousCommitmentRuntime)<mro.index(_StoppingDispatch)<mro.index(PacedMultirateController)
