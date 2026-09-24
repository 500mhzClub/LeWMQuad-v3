import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.relative_pose_uncertainty_development import relative_point_moments,shared_rigid_error_jacobian


def poses():
    return np.array([1.,.3,.05]),rotation_increment([.02,.03,.4]),np.array([.2,-.1,.01]),rotation_increment([-.02,.01,.1])


def test_exact_rigid_reference_change_preserves_relative_points():
    pc,rc,pf,rf=poses(); q=np.array([[.3,-.4,.1],[1.,.5,-.2]])
    baseline=relative_point_moments(q,pc,rc,pf,rf,np.zeros((12,12)))
    rotation=rotation_increment([.5,-.2,1.]); translation=np.array([30,-20,10.])
    changed=relative_point_moments(q,rotation@pc+translation,rotation@rc,
        rotation@pf+translation,rotation@rf,np.zeros((12,12)))
    np.testing.assert_allclose(changed['point_mean_stored_body_m'],baseline['point_mean_stored_body_m'],atol=1e-13)


def test_complete_shared_gauge_covariance_cancels_including_cross_terms():
    pc,rc,pf,rf=poses(); q=np.array([[.3,-.4,.1],[1.,.5,-.2]])
    gauge=shared_rigid_error_jacobian(pc,pf); cov=gauge@np.diag([.1,.2,.3,.02,.03,.04])@gauge.T
    result=relative_point_moments(q,pc,rc,pf,rf,cov)
    np.testing.assert_allclose(result['jacobian']@gauge,0.,atol=1e-15)
    np.testing.assert_allclose(result['point_covariance_m2'],0.,atol=1e-15)
    assert not result['joint_input_calibration_established'] and not result['navigation_qualified']


@pytest.mark.parametrize('correlation,expected',[(1.,0.),(0.,2.),(-1.,4.)])
def test_equal_marginal_variance_is_not_enough_to_determine_relative_error(correlation,expected):
    cov=np.zeros((12,12)); cov[0,0]=cov[6,6]=1.; cov[0,6]=cov[6,0]=correlation
    result=relative_point_moments([[1,0,0]],np.zeros(3),np.eye(3),np.zeros(3),np.eye(3),cov)
    assert result['point_covariance_m2'][0,0,0]==pytest.approx(expected)


def test_joint_error_jacobian_matches_independent_finite_differences():
    pc,rc,pf,rf=poses(); q=np.array([[.3,-.4,.1]]); cov=np.eye(12)*1e-6
    result=relative_point_moments(q,pc,rc,pf,rf,cov)
    for column in range(12):
        values=[]
        for sign in (-1,1):
            error=np.zeros(12); error[column]=sign*1e-6
            current_r=rotation_increment(error[3:6])@rc; stored_r=rotation_increment(error[9:12])@rf
            values.append(stored_r.T@(current_r@q[0]+pc+error[:3]-pf-error[6:9]))
        numerical=(values[1]-values[0])/2e-6
        np.testing.assert_allclose(numerical,result['jacobian'][0,:,column],atol=2e-10)


@pytest.mark.parametrize('fault',['missing','marginals_only','nonsymmetric','indefinite','nan'])
def test_missing_or_invalid_joint_error_model_cannot_approve_a_transport(fault):
    pc,rc,pf,rf=poses(); cov=np.eye(12)
    if fault=='missing': cov=None
    elif fault=='marginals_only': cov=np.ones(12)
    elif fault=='nonsymmetric': cov[0,6]=.5
    elif fault=='indefinite': cov[0,6]=cov[6,0]=2.
    elif fault=='nan': cov[0,0]=np.nan
    with pytest.raises(SensorContractError): relative_point_moments([[1,0,0]],pc,rc,pf,rf,cov)
