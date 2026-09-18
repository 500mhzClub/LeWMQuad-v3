import numpy as np
import torch

from lewm.delayed_action_planning_development import (
    delayed_candidate_inputs, score_delayed_predictions, ScheduledCommand)
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.observation_horizon_plan_development import validate_plan


def test_known_prefix_candidate_interval_and_explicit_zero_tail():
    history = {k:torch.zeros(s) for k,s in dict(rgb=(4,3,96,128),body=(4,20,63),control=(4,15,7)).items()}
    prefix = [[.2,0,0],[0,0,-.45]]
    x = delayed_candidate_inputs(history,prefix)
    blocks, valid = x['known_action_blocks'],x['known_action_valid']
    _,offsets = validate_plan(blocks,valid,6)
    torch.testing.assert_close(offsets[0],torch.arange(1,9)*100_000_000)
    decoded = blocks*torch.tensor([.3,1.,.5])
    for i,action in enumerate(ACTIONS):
        torch.testing.assert_close(decoded[i,:2,0],torch.tensor(prefix))
        torch.testing.assert_close(decoded[i,2,0],torch.tensor(candidate_commands(action)[0]))
    assert not torch.count_nonzero(decoded[:,3:])


def test_score_uses_motion_during_delayed_commit_not_later_tail():
    p = np.zeros((6,8,5));p[:,:,4]=-10.
    p[1,2,0]=.1;p[1,3:,0]=.1
    p[2,7,0]=100.
    r=score_delayed_predictions(p,[1.,0.])
    assert r['action']=='forward' and r['scoring_endpoint_offset_ns']==300_000_000
    assert r['common_prefix_max_absolute_spread']==[0.]*5


def test_missed_deadline_cannot_be_retimed_and_command_expires_or_is_vetoed():
    assert ScheduledCommand.prepare('forward',observed_ns=0,completed_ns=200_000_001) is None
    plan=ScheduledCommand.prepare('forward',observed_ns=0,completed_ns=150_000_000)
    for now in (0,199_999_999,300_000_000,900_000_000):
        assert plan.request(now_ns=now,fresh_observation_allows_motion=True)==[0.,0.,0.]
    assert plan.request(now_ns=200_000_000,fresh_observation_allows_motion=True)==[.2,0.,0.]
    assert plan.request(now_ns=200_000_000,fresh_observation_allows_motion=False)==[0.,0.,0.]
