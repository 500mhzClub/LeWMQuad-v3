from copy import deepcopy
import math
import numpy as np
import pytest
from lewm.commitment_pose_waypoint_utility_development import potential, score_commitment_pose
from lewm.geometry_progress_predictive_selection_development import score_candidates


def fixture(goal):
    p=np.zeros((6,8,5));p[:,:,3]=1.;p[:,:,4]=-10.
    p[1,:,:2]=[.1,0]
    for i,yaw in ((4,math.pi/4),(5,-math.pi/4)):
        p[i,:,2]=math.sin(yaw);p[i,:,3]=math.cos(yaw)
    s=score_candidates(p,goal_body_xy_m=goal,contact_penalty_m=1.2)
    return s|dict(prediction=p.tolist())


@pytest.mark.parametrize('goal,action',[([0,1],'left_turn'),([0,-1],'right_turn'),([1,0],'forward')])
def test_turn_alignment_then_forward_progress(goal,action):
    source=fixture(goal);before=deepcopy(source)
    row=score_commitment_pose(source)
    assert row['action']==action and source==before and row['prediction']==source['prediction']


def test_contact_uses_only_committed_horizon_and_keeps_all_predictions():
    source=fixture([1,0]);source['prediction'][1][-1][4]=10.
    assert score_commitment_pose(source)['action']=='forward'
    for horizon in source['prediction'][1]:horizon[4]=10.
    row=score_commitment_pose(source)
    assert row['action']=='hold' and row['surface_conflict_filter_still_required']


def test_near_target_alignment_fades_and_undefined_yaw_rejected():
    assert potential([0,0],math.pi)==(0.,0.,0.)
    assert potential([1e-9,0],math.pi)[0]<4e-9
    source=fixture([1,0]);source['prediction'][0][0][2:4]=[0.,0.]
    with pytest.raises(ValueError):score_commitment_pose(source)
