from threading import Event,Thread
import time
import numpy as np

from lewm.measured_latency_simulation_development import MeasuredLatencyClock
from lewm.delayed_action_planning_development import ScheduledCommand,score_delayed_predictions


def test_measured_work_cannot_publish_before_simulation_reaches_its_cost():
    clock=MeasuredLatencyClock();started=Event();done=Event()
    def work():
        clock.begin('tracking');time.sleep(.002);started.set();clock();clock.end();done.set()
    thread=Thread(target=work);thread.start()
    try:
        assert started.wait(1) and not done.wait(.01)
        clock.advance(1_600_000_000)
        assert done.wait(1)
        row=clock.releases[0]
        assert row['measured_service_ns']>0
        assert row['released_ns']>=row['earliest_release_ns']>row['start_sim_ns']
    finally:clock.close();thread.join(timeout=1)


def test_300ms_delay_scores_only_the_300_to_400ms_command():
    p=np.zeros((6,8,5));p[:,:,4]=-10
    p[1,2,0]=.5;p[1,3,0]=.5  # prior motion gives no candidate-interval progress
    p[2,3,0]=.1
    result=score_delayed_predictions(p,[1.,0.],delay_ticks=3)
    assert result['action_index']==2 and result['dispatch_offset_ns']==300_000_000
    assert result['scoring_endpoint_offset_ns']==400_000_000
    plan=ScheduledCommand.prepare('forward',observed_ns=0,completed_ns=250_000_000,delay_ticks=3)
    assert plan.dispatch_ns==300_000_000 and plan.expires_ns==400_000_000
    assert ScheduledCommand.prepare('forward',observed_ns=0,completed_ns=300_000_001,delay_ticks=3) is None
