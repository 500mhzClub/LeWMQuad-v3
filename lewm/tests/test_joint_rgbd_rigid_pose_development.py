from copy import deepcopy

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.joint_rgbd_rigid_pose_development import fit,register,angle,scatter,RigidRGBDKeyframePose
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.rgbd_correspondence_motion_development import project
from lewm.tests.test_rgbd_correspondence_motion_development import point_fixture,texture,packets


def test_planar_noncollinear_points_determine_proper_rigid_pose():
    a,b,ua,ub,R,t=point_fixture();estimated,position,quality=fit(a,b)
    np.testing.assert_allclose(estimated,R,atol=1e-12)
    np.testing.assert_allclose(position,t,atol=1e-12)
    assert quality['reference_scatter_rms_m'][2]<1e-12  # A plane is not a line.


@pytest.mark.parametrize('mode',['joint','gyro'])
def test_matched_consensus_recovers_transform_with_outliers(mode):
    a,b,ua,ub,R,t=point_fixture();b=b.copy();b[:3]+=[.1,.2,.3]
    estimate,p,mask,quality=register(a,b,ua,ub,gyro_rotation=R,mode=mode,frame=1)
    np.testing.assert_allclose(estimate,R,atol=1e-12);np.testing.assert_allclose(p,t,atol=1e-12)
    assert not mask[:3].any() and mask[3:].all() and quality['matched_consensus_rules']


def test_joint_pose_uses_rgbd_rotation_not_biased_gyro():
    a,b,ua,ub,R,t=point_fixture();G=R@rotation_increment([0,0,.002])
    estimated,p,_,q=register(a,b,ua,ub,gyro_rotation=G,mode='joint',frame=2)
    np.testing.assert_allclose(estimated,R,atol=1e-12);np.testing.assert_allclose(p,t,atol=1e-12)
    assert q['gyro_disagreement_rad']==pytest.approx(.002)
    fixed,_,_,_=register(a,b,ua,ub,gyro_rotation=G,mode='gyro',frame=2)
    np.testing.assert_allclose(fixed,G,atol=1e-12)


@pytest.mark.parametrize('points',[np.zeros((10,3)),np.column_stack((np.arange(10),np.zeros((10,2)))),np.zeros((2,3))])
def test_degenerate_or_missing_scatter_is_rejected(points):
    with pytest.raises(SensorContractError):scatter(points)


def test_large_cross_sensor_rotation_disagreement_is_rejected():
    a,b,ua,ub,R,_=point_fixture()
    with pytest.raises(SensorContractError,match='disagree'):
        register(a,b,ua,ub,gyro_rotation=R@rotation_increment([0,0,.2]),mode='joint',frame=1)


@pytest.mark.parametrize('fault',['few','coverage','wrong_pixels'])
def test_rigid_registration_does_not_invent_support(fault):
    a,b,ua,ub,R,_=point_fixture()
    if fault=='few':a,b,ua,ub=a[:3],b[:3],ua[:3],ub[:3]
    elif fault=='coverage':ua[:]=[80,80];ub[:]=[80,80]
    else:ub+=20
    with pytest.raises(SensorContractError):register(a,b,ua,ub,gyro_rotation=R,mode='joint',frame=1)


@pytest.mark.parametrize('mode',['joint','gyro'])
def test_actual_sensor_packets_preserve_pose_parent_history(monkeypatch,mode):
    monkeypatch.setattr('lewm.joint_rgbd_rigid_pose_development.support_near_limit',lambda r:True)
    model=RigidRGBDKeyframePose(mode)
    rows=[model.observe(*r[:3],now_ns=r[3]) for r in packets([texture()]*3)]
    assert [n['parent_frame'] for n in model.nodes]==[None,0,1]
    assert rows[-1]['position_error_bound'] is None and rows[-1]['orientation_error_bound'] is None
    assert not rows[-1]['global_history_reset'] and not rows[-1]['navigation_qualified']


@pytest.mark.parametrize('fault',['blank','clock','privileged'])
def test_terminal_failure_never_reanchors(fault):
    import hashlib
    rows=list(packets([texture()]*3));model=RigidRGBDKeyframePose('joint');model.observe(*rows[0][:3],now_ns=rows[0][3])
    p,d,f,now=deepcopy(rows[1])
    if fault=='blank':p['image']['rgb'][:]=128;d['rgb_sha256']=hashlib.sha256(p['image']['rgb'].tobytes()).hexdigest()
    elif fault=='clock':now+=100_000_000
    else:p['native_pose']=[0,0,0]
    with pytest.raises(SensorContractError):model.observe(p,d,f,now_ns=now)
    with pytest.raises(SensorContractError):model.observe(*rows[2][:3],now_ns=rows[2][3])
    assert len(model.nodes)==1 and model.failed
