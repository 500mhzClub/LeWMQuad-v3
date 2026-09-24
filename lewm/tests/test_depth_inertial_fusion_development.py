"""Analytic kernel and actual observer contracts; no navigation evidence."""
from copy import deepcopy
import hashlib

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.depth_inertial_fusion_development import WeakSubspaceIntegrator, DepthInertialState
from lewm.tests.test_observed_traversal_controller_development import Stream
from lewm.tests.test_depth_local_surfaces_development import packet
from lewm.relative_gyro_turn_development import rotation_increment


def state(policy, delta=None, weak=(), rotation=None):
    """Synthetic registered constraints, not synthetic sensor observability."""
    rotation=np.eye(3) if rotation is None else rotation
    w=np.asarray(weak,dtype=float).reshape(-1,3)
    rank=3-len(w)
    motion=None if delta is None else {
        'rank':rank, 'status':'OBSERVED_TRANSLATION' if rank==3 else 'PARTIALLY_OBSERVED_TRANSLATION',
        'translation_previous_body_m':list(delta) if rank==3 else None,
        'observable_projection_previous_body_m':((np.eye(3)-w.T@w)@delta).tolist(),
        'weak_directions_previous_body':w.tolist()}
    now=policy['sensor_state']['decision_ns']
    return {'measured_ns':now, 'motion':motion,
        'local_surfaces':{'measured_ns':now,'identity':(0,0,0),
            'rgb_sha256':hashlib.sha256(policy['image']['rgb'].tobytes()).hexdigest()},
        'relative_orientation':{'decision_ns':now,
            'rotation_initial_body_from_current_body':rotation.tolist()}}


def frame(stream,tick,acceleration=(0.,0.,0.)):
    policy=stream.frame(tick)[0]
    force=policy['sensor_state']['sensed']['specific_force']
    # Acceleration begins after the initial anchor; assignment is a function
    # of absolute sample time so overlapping histories remain immutable.
    selected=force['measured_ns']>1_600_000_000
    force['values'][selected]+=np.asarray(acceleration)
    return policy


def initialized():
    stream=Stream(); model=WeakSubspaceIntegrator(); p=frame(stream,0)
    first=model.observe(p,state(p))
    assert first['position_initial_body_m']==[0.,0.,0.]
    assert first['velocity_initial_body_m_s'] is None
    return stream,model


@pytest.mark.parametrize('weak', [[[1,0,0]],[[0,1,0]],[[0,0,1]],[[1,0,0],[0,1,0]]])
def test_constant_velocity_fills_only_weak_subspace_and_preserves_evidence(weak):
    stream,model=initialized(); delta=np.array([.02,-.01,.003])
    for tick in range(1,8):
        p=frame(stream,tick); s=state(p,delta,weak if tick>1 else ())
        before=deepcopy(s); result=model.observe(p,s)
        assert s==before
        np.testing.assert_allclose(result['translation_previous_body_m'],delta,atol=1e-12)
        np.testing.assert_allclose(result['position_initial_body_m'],tick*delta,atol=1e-12)
    assert result['kind']=='INERTIALLY_PREDICTED_WEAK_COMPONENT'
    assert result['depth_rank']==3-len(weak)
    assert s['motion']['translation_previous_body_m'] is None


def test_constant_acceleration_and_observed_velocity_initialization():
    stream,model=initialized(); acceleration=np.array([.05,-.03,.01])
    velocity=np.array([.1,.02,0.]); position=np.zeros(3)
    for tick in range(1,11):
        delta=.1*velocity+.005*acceleration; position+=delta; velocity+=.1*acceleration
        p=frame(stream,tick,acceleration)
        result=model.observe(p,state(p,delta,[[1,0,0],[0,1,0]] if tick>1 else ()))
        np.testing.assert_allclose(result['position_initial_body_m'],position,atol=1e-12)
        np.testing.assert_allclose(result['velocity_initial_body_m_s'],velocity,atol=1e-12)


def test_projectors_use_previous_body_frame_during_rotation():
    stream,model=initialized(); previous=np.eye(3); velocity=np.array([.12,.03,.01])
    for tick in range(1,9):
        rotation=rotation_increment([0,0,.2*tick])
        p=frame(stream,tick); body_delta=previous.T@(.1*velocity)
        result=model.observe(p,state(p,body_delta,[[1,0,0]] if tick>1 else (),rotation))
        np.testing.assert_allclose(result['position_initial_body_m'],.1*tick*velocity,atol=1e-12)
        previous=rotation


def test_long_weak_interval_exhausts_budget_and_recovery_does_not_erase_uncertainty():
    stream,model=initialized(); scales=[]
    for tick in range(1,31):
        p=frame(stream,tick)
        result=model.observe(p,state(p,[.01,0,0],[[1,0,0]] if 1<tick<30 else ()))
        scales.append(result['position_error_scale_m'])
    assert np.all(np.diff(scales)>0)
    assert not result['usable_under_declared_proxy_budget']
    assert result['depth_rank']==3 and result['consecutive_weak_seconds']==0
    assert result['weak_intervals']==28
    assert not result['assumptions']['calibrated_covariance']
    assert not result['assumptions']['hardware_qualified']


def test_out_of_assumption_bias_is_not_misrepresented_as_a_guaranteed_bound():
    stream,model=initialized()
    # True constant velocity; an unmodelled 0.2 m/s^2 sensor bias exceeds
    # the declared 0.02 assumption. The proxy must not be called calibrated.
    for tick in range(1,12):
        p=frame(stream,tick,[.2,0,0])
        result=model.observe(p,state(p,[.01,0,0],[[1,0,0]] if tick>1 else ()))
    actual_error=abs(result['position_initial_body_m'][0]-.11)
    assert actual_error>result['position_error_scale_m']
    assert result['usable_under_declared_proxy_budget']
    assert result['assumptions']['calibrated_covariance'] is False


@pytest.mark.parametrize('fault', ['force_rewrite','gyro_rewrite','force_invalid','force_nan',
    'clock','identity','rgb_identity','orientation','rank','basis','projection','full_projection',
    'privilege','envelope'])
def test_invalid_input_latches(fault):
    stream,model=initialized(); p=frame(stream,1); s=state(p,[.01,0,0])
    if fault=='force_rewrite': p['sensor_state']['sensed']['specific_force']['values'][0,0]=.1
    elif fault=='gyro_rewrite': p['sensor_state']['sensed']['gyro']['values'][0,0]=.1
    elif fault=='force_invalid': p['sensor_state']['sensed']['specific_force']['valid'][-1]=False
    elif fault=='force_nan': p['sensor_state']['sensed']['specific_force']['values'][-1,0]=np.nan
    elif fault=='clock': s['measured_ns']+=1
    elif fault=='identity': s['local_surfaces']['identity']=(0,0,1)
    elif fault=='rgb_identity': s['local_surfaces']['rgb_sha256']='0'*64
    elif fault=='orientation': s['relative_orientation']['rotation_initial_body_from_current_body'][0][0]=2
    elif fault=='rank': s['motion']['rank']=True
    elif fault=='basis': s=state(p,[.01,0,0],[[2,0,0]])
    elif fault=='projection': s=state(p,[.01,0,0],[[1,0,0]]); s['motion']['observable_projection_previous_body_m']=[.01,0,0]
    elif fault=='full_projection': s['motion']['translation_previous_body_m']=[0,0,0]
    elif fault=='privilege': p['world_pose']=[0,0,0]
    elif fault=='envelope': s=state(p,[.2,0,0])
    with pytest.raises(SensorContractError): model.observe(p,s)
    assert model.failed
    with pytest.raises(SensorContractError,match='latched'): model.observe(p,state(p,[.01,0,0]))


def test_weak_motion_without_observed_velocity_is_rejected():
    stream,model=initialized(); p=frame(stream,1)
    with pytest.raises(SensorContractError): model.observe(p,state(p,[.01,0,0],[[1,0,0]]))


def test_live_wrapper_preserves_rank_one_failure_without_inventing_a_velocity():
    stream=Stream(); model=DepthInertialState()
    p,fast,now=stream.frame(0); _,depth,_=packet('front',tick=0)
    result=model.observe(p,depth,fast,now_ns=now)
    assert result['depth_state']['motion'] is None
    assert not result['navigation_qualified']
    p,fast,now=stream.frame(1); _,depth,_=packet('front',tick=1)
    with pytest.raises(SensorContractError): model.observe(p,depth,fast,now_ns=now)
    assert model.integrator.failed
