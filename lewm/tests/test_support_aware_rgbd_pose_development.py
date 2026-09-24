from copy import deepcopy
import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.support_aware_rgbd_pose_development import SupportAwareRGBDPose,RobustReferenceEvidence,contaminated_box,support_near_limit
from lewm.tests.test_rgbd_correspondence_motion_development import texture,packets


@pytest.mark.parametrize('count,wanted',[(6,True),(7,True),(8,False),(12,False)])
def test_support_margin_preserves_six_cell_acceptance(count,wanted):
    assert support_near_limit(dict(reference_grid_cells=count,current_grid_cells=12))==wanted


def test_failed_support_cannot_trigger_reanchoring():
    with pytest.raises(SensorContractError):support_near_limit(dict(reference_grid_cells=5,current_grid_cells=8))


@pytest.mark.parametrize('bad',[0,1,2])
def test_contamination_enclosure_contains_truth_under_correlated_good_errors(bad):
    truth=np.array([.1,-.2,.3]);d=np.tile(truth+[.003,-.002,.001],(11,1));r=np.full(11,.004)
    d[:bad]=[20.,-30.,40.]
    b=contaminated_box(d,r,maximum_outliers=2)
    assert b['status']=='CONDITIONAL_CONTAMINATED_BOX'
    assert np.all(np.asarray(b['lower'])<=truth) and np.all(truth<=np.asarray(b['upper']))


def test_excess_corruption_and_disjoint_constraints_are_not_a_consensus_certificate():
    d=np.array([[0.,0,0],[10.,10,10],[20.,20,20]])
    b=contaminated_box(d,np.full(3,.1),maximum_outliers=0)
    assert b['status']=='INCONSISTENT_CONTAMINATION_HYPOTHESIS' and not b['joint_consensus_proven']


@pytest.mark.parametrize('bad',[True,-1,3])
def test_invalid_contamination_budget_rejected(bad):
    with pytest.raises(SensorContractError):contaminated_box(np.zeros((5,3)),np.ones(5),maximum_outliers=bad)


def test_forced_accepted_promotion_keeps_pose_error_and_parent_chain(monkeypatch):
    # Synthetic predicate intervention only; production has the separate tested margin rule.
    monkeypatch.setattr('lewm.support_aware_rgbd_pose_development.support_near_limit',lambda r:True)
    model=SupportAwareRGBDPose();evidence=RobustReferenceEvidence();rows=[]
    for p,d,f,now in packets([texture()]*4):
        row=model.observe(p,d,f,now_ns=now);row['robust']=evidence.observe(row);rows.append(row)
    assert [n['parent_frame'] for n in model.nodes]==[None,0,1,2]
    assert all(r['promoted_keyframe'] for r in rows[1:])
    assert rows[3]['conditional_global_position_radius_m']>rows[1]['conditional_global_position_radius_m']>0
    assert rows[3]['robust']['global_radius_m']>rows[1]['robust']['global_radius_m']>0
    assert all(not r['global_history_reset'] and not r['navigation_qualified'] for r in rows)


def test_unknown_promoted_robust_anchor_is_never_reset(monkeypatch):
    monkeypatch.setattr('lewm.support_aware_rgbd_pose_development.support_near_limit',lambda r:True)
    model=SupportAwareRGBDPose();evidence=RobustReferenceEvidence();rows=[]
    for p,d,f,now in packets([texture()]*3):rows.append(model.observe(p,d,f,now_ns=now))
    evidence.observe(rows[0])
    original=rows[1];bad=deepcopy(original);a=np.asarray(bad['registration']['reference_inlier_points_body_m'])
    a[:,0]+=np.arange(len(a))*.1;bad['registration']['reference_inlier_points_body_m']=a.tolist()
    assert evidence.observe(bad)['global_radius_m'] is None
    assert evidence.observe(rows[2])['global_radius_m'] is None
