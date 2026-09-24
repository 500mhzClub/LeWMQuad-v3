import numpy as np
import pytest

from lewm.foot_load_sensor_development import IdealFootForceSample
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.simulated_body_observation_development import CALIBRATION
from lewm.support_kinematics_development import FootJacobians,CausalQuietUp,predict_support_motion
from scripts.analyze_go2_ground_plane_development_v1 import URDF


@pytest.fixture
def kinematics(): return FootJacobians(URDF)


def inputs():
    body=dict(identity='test',calibration_id=CALIBRATION,measured_ns=20_000_000,available_ns=20_000_000,
        q=np.tile([0,.7,-1.4],4).reshape(4,3).T.ravel(),dq=np.zeros(12),gyro=np.zeros(3),specific_force=[0,0,9.81],valid=True)
    loads=[IdealFootForceSample('test',t,t,np.tile([0,0,20.],(4,1)),np.ones(4,bool),np.zeros(4,bool)) for t in range(0,22_000_000,2_000_000)]
    return body,loads


@pytest.mark.parametrize('seed',[1,2,3,4])
def test_all_linear_jacobian_columns_central_difference(kinematics,seed):
    q=np.random.default_rng(seed).uniform(-1.2,1.2,12); d=kinematics.calculate(q); h=1e-6
    for i in range(12):
        step=np.eye(12)[i]*h
        derivative=(kinematics.calculate(q+step)['position_body_m']-kinematics.calculate(q-step)['position_body_m'])/(2*h)
        np.testing.assert_allclose(d['linear_jacobian'][:,:,i],derivative,atol=2e-10,rtol=0)


def test_angular_jacobian_and_disjoint_leg_columns(kinematics):
    q=np.linspace(-.6,.6,12); d=kinematics.calculate(q); h=1e-6
    for i in range(12):
        plus=kinematics.calculate(q+np.eye(12)[i]*h)['rotation_body_from_foot']
        minus=kinematics.calculate(q-np.eye(12)[i]*h)['rotation_body_from_foot']
        W=((plus-minus)/(2*h))@d['rotation_body_from_foot'].transpose(0,2,1)
        np.testing.assert_allclose(d['angular_jacobian'][:,:,i],W[:,[2,0,1],[1,2,0]],atol=2e-10,rtol=0)
        for foot in range(4):
            if i%4!=foot: assert not d['linear_jacobian'][foot,:,i].any()


def test_stationary_and_rolling_sign(kinematics):
    body,loads=inputs(); body['gyro']=np.array([0,1.,0])
    r=predict_support_motion(kinematics,body,loads,identity='test',up_body=[0,0,1])
    a=np.array(r['modes']['stationary_centre']['per_foot_velocity_body_m_s'])
    b=np.array(r['modes']['level_sphere_rolling']['per_foot_velocity_body_m_s'])
    np.testing.assert_allclose(b-a,np.tile([.022,0,0],(4,1)),atol=1e-14)
    np.testing.assert_allclose(a,-np.cross(body['gyro'],np.array(r['foot_position_body_m'])))
    assert not r['support_established'] and r['physical_velocity_error_bound_m_s'] is None


def test_zero_motion_not_slip_proof(kinematics):
    b,l=inputs(); r=predict_support_motion(kinematics,b,l,identity='test',up_body=[0,0,1])
    assert r['modes']['stationary_centre']['consensus_velocity_body_m_s']==[0,0,0]
    assert not r['slip_excluded']  # Identical sensors can coexist with constant-velocity common slip.


def test_missing_load_not_selected(kinematics):
    b,l=inputs(); l[-1]=IdealFootForceSample('test',20_000_000,20_000_000,np.full((4,3),np.nan),np.zeros(4,bool),np.zeros(4,bool))
    r=predict_support_motion(kinematics,b,l,identity='test',up_body=[0,0,1])
    assert not any(r['selected_loaded_feet'])
    assert r['modes']['stationary_centre']['consensus_velocity_body_m_s'] is None


@pytest.mark.parametrize('fault',['identity','calibration_id','clock','native_pose','invalid','loadgap','up'])
def test_strict_causal_boundary(kinematics,fault):
    b,l=inputs(); up=[0,0,1]
    if fault in ('identity','calibration_id'): b[fault]='wrong'
    elif fault=='clock': b['available_ns']+=1
    elif fault=='native_pose': b['base_pose_world']=[0]*7
    elif fault=='invalid': b['valid']=False
    elif fault=='loadgap': l=l[:-1]
    else: up=[0,0,2]
    with pytest.raises(ValueError): predict_support_motion(kinematics,b,l,identity='test',up_body=up)


def test_rotating_quiet_up_co_rotates_accelerometer():
    m=CausalQuietUp(); rate=np.array([.2,-.1,.4]); initial=np.array([0,0,1.]); last=None
    for i in range(201):
        stamp=1_300_000_000+i*2_000_000; R=rotation_increment(rate*i*.002)
        a=R.T@initial*9.81 if i%10==0 else None
        up=m.update(stamp,rate,a)
        if i<100: assert up is None
        else: np.testing.assert_allclose(up,R.T@initial,atol=1e-12)


def test_up_requires_complete_initialization_and_order():
    m=CausalQuietUp()
    with pytest.raises(ValueError): m.update(1_300_000_001,[0,0,0])
    m.update(1_300_000_000,[0,0,0],[0,0,9.81])
    with pytest.raises(ValueError): m.update(1_304_000_000,[0,0,0])
