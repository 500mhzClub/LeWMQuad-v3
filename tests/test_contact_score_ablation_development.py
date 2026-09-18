import numpy as np
from lewm.contact_score_ablation_development import contact_score_prediction
from lewm.waypoint_alignment_planning_development import score_waypoint_alignment


def test_disabled_contact_changes_ranking_without_changing_motion_or_input():
    p=np.zeros((6,8,5));p[:,:,3]=1.;p[:,:,4]=-8.
    p[1,:,0]=np.arange(1,9)*.02;p[1,:,4]=3.
    original=p.copy();changed=p.copy();changed[:,:,4]+=100
    disabled=contact_score_prediction(p,'disabled')
    np.testing.assert_array_equal(disabled,contact_score_prediction(changed,'disabled'))
    np.testing.assert_array_equal(disabled[:,:,:4],p[:,:,:4])
    np.testing.assert_array_equal(p,original)
    np.testing.assert_array_equal(contact_score_prediction(p,'learned'),p)
    learned=score_waypoint_alignment(p,[1.,0.],delay_ticks=3,commit_ticks=4)
    control=score_waypoint_alignment(disabled,[1.,0.],delay_ticks=3,commit_ticks=4)
    assert learned['action']=='hold' and control['action']=='forward'
    assert all(r['predicted_contact_by_commit_end']==0. for r in control['candidates'])
