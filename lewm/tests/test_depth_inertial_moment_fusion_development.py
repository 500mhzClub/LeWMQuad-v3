"""Time-resolved synthetic mechanics and inherited causal fault contracts."""
import numpy as np
import pytest

import lewm.tests.test_depth_inertial_fusion_development as original
from lewm.depth_inertial_moment_fusion_development import (
    interval_moments, MomentWeakSubspaceIntegrator, MomentDepthInertialState)
from lewm.depth_inertial_fusion_development import WeakSubspaceIntegrator as OriginalIntegrator
from lewm.causal_sensor_state import SensorContractError
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.tests.test_observed_traversal_controller_development import Stream
from lewm.tests.test_depth_inertial_fusion_development import (
    test_constant_velocity_fills_only_weak_subspace_and_preserves_evidence,
    test_constant_acceleration_and_observed_velocity_initialization,
    test_projectors_use_previous_body_frame_during_rotation,
    test_long_weak_interval_exhausts_budget_and_recovery_does_not_erase_uncertainty,
    test_out_of_assumption_bias_is_not_misrepresented_as_a_guaranteed_bound,
    test_invalid_input_latches,
    test_weak_motion_without_observed_velocity_is_rejected,
    test_live_wrapper_preserves_rank_one_failure_without_inventing_a_velocity)


@pytest.fixture(autouse=True)
def inherited_contracts_use_successor(monkeypatch):
    # Test-only injection; original files/results are never changed. Pytest
    # restores these bindings after each case, including imported contracts.
    monkeypatch.setattr(original,'WeakSubspaceIntegrator',MomentWeakSubspaceIntegrator)
    monkeypatch.setattr(original,'DepthInertialState',MomentDepthInertialState)


def fine_integrate(acceleration,velocity):
    """100 independent substeps per bin, with no moment weights."""
    velocity=np.asarray(velocity,dtype=float).copy(); displacement=np.zeros(3)
    for a in acceleration:
        for _ in range(100):
            displacement+=velocity*.0002+.5*np.asarray(a)*.0002**2
            velocity+=np.asarray(a)*.0002
    return displacement,velocity


@pytest.mark.parametrize('acceleration',[
    [[1,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
    [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[1,0,0]],
    [[.3,-.2,.4],[-.2,.5,-.4],[0,0,0],[.9,-.8,.1],[-1,.5,.2]],
    [[0,0,0]]*5])
def test_moments_match_independently_integrated_fine_time_reference(acceleration):
    initial=np.array([.1,-.03,.02]); d,v=fine_integrate(acceleration,initial)
    moments=interval_moments(acceleration)
    np.testing.assert_allclose(d,.1*initial+moments['displacement_from_initial_velocity_m'],atol=1e-13)
    np.testing.assert_allclose(v,initial+moments['velocity_increment_m_s'],atol=1e-13)
    np.testing.assert_allclose(v,d/.1+moments['endpoint_velocity_correction_m_s'],atol=1e-13)


def test_equal_mean_early_late_acceleration_have_different_moments():
    early=np.zeros((5,3)); early[0,0]=1
    late=early[::-1].copy(); a,b=interval_moments(early),interval_moments(late)
    assert a['velocity_increment_m_s']==b['velocity_increment_m_s']
    assert a['displacement_from_initial_velocity_m'][0]>b['displacement_from_initial_velocity_m'][0]
    assert a['endpoint_velocity_correction_m_s'][0]<b['endpoint_velocity_correction_m_s'][0]


def acceleration_bin(index):
    if index<=0: return np.zeros(3)
    return np.array([.4,-.2,.1,-.4,.1])[(index-1)%5]*np.array([1.,-.5,.25])


def moving_packet(stream,tick,rate):
    p=stream.frame(tick)[0]
    sensed=p['sensor_state']['sensed']; f=sensed['specific_force']; g=sensed['gyro']
    for i,ns in enumerate(f['measured_ns']):
        offset=int(ns)-1_600_000_000
        rotation=rotation_increment(rate*max(0,offset)*1e-9)
        f['values'][i]=rotation.T@(acceleration_bin(offset//20_000_000)+[0.,0.,9.81])
        g['values'][i]=rate if offset>0 else np.zeros(3)
    return p


@pytest.mark.parametrize('rate', [[0,0,0],[.07,-.04,.2]])
def test_twelve_seconds_switched_weak_subspaces_and_rotating_gravity(rate):
    rate=np.asarray(rate,dtype=float); stream=Stream(); model=MomentWeakSubspaceIntegrator()
    previous=np.eye(3); velocity=np.array([.12,-.02,.005]); position=np.zeros(3)
    last_scale=0.; budget_exhausted=False
    for tick in range(121):
        p=moving_packet(stream,tick,rate); rotation=rotation_increment(rate*.1*tick)
        if tick:
            a=np.array([acceleration_bin(5*(tick-1)+i) for i in range(1,6)])
            delta,velocity=fine_integrate(a,velocity); position+=delta
            weak=() if tick in (1,30,60,90,120) else ([[1,0,0],[0,1,0]] if tick%3 else [[0,0,1]])
            observed=original.state(p,previous.T@delta,weak,rotation)
        else: observed=original.state(p)
        result=model.observe(p,observed)
        np.testing.assert_allclose(result['position_initial_body_m'],position,atol=1e-11)
        if tick: np.testing.assert_allclose(result['velocity_initial_body_m_s'],velocity,atol=1e-11)
        assert result['position_error_scale_m']>=last_scale; last_scale=result['position_error_scale_m']
        budget_exhausted|=not result['usable_under_declared_proxy_budget']
        assert result['uncertainty_model_validated'] is False
        previous=rotation
    assert budget_exhausted  # Perfect nominal integration does not erase its proxy budget.


def test_v1_average_acceleration_loses_endpoint_velocity_information():
    stream=Stream(); old=OriginalIntegrator(); new=MomentWeakSubspaceIntegrator()
    p=moving_packet(stream,0,np.zeros(3)); s=original.state(p)
    old.observe(p,s); new.observe(p,s)
    acceleration=np.array([acceleration_bin(i) for i in range(1,6)])
    d,v=fine_integrate(acceleration,[.1,0,0])
    p=moving_packet(stream,1,np.zeros(3)); s=original.state(p,d)
    first=old.observe(p,s); second=new.observe(p,s)
    assert np.linalg.norm(np.array(first['velocity_initial_body_m_s'])-v)>.001
    np.testing.assert_allclose(second['velocity_initial_body_m_s'],v,atol=1e-12)


def test_unobserved_intra_bin_jerk_is_not_recovered_by_bin_moments():
    # Every 20-ms bin has zero mean acceleration; fine-time sinusoidal motion
    # nevertheless accumulates displacement. Retain this sampling limitation.
    velocity=0.; displacement=0.; dt=.00002
    for k in range(5000):
        acceleration=np.sin(2*np.pi*((k+.5)*dt)/.02)
        displacement+=velocity*dt+.5*acceleration*dt*dt
        velocity+=acceleration*dt
    moments=interval_moments(np.zeros((5,3)))
    assert abs(velocity)<1e-12 and displacement>.0003
    assert moments['displacement_from_initial_velocity_m']==[0.,0.,0.]
    assert not moments['intra_bin_jerk_bounded']


@pytest.mark.parametrize('acceleration',[np.zeros((4,3)),np.zeros((5,2)),np.full((5,3),np.nan)])
def test_moment_shape_and_finiteness_faults(acceleration):
    with pytest.raises(SensorContractError): interval_moments(acceleration)


def test_stale_endpoint_force_cannot_be_used_as_current_interval():
    p=Stream().frame(0)[0]
    for name in ('gyro','specific_force'):
        for field in ('measured_ns','available_ns'): p['sensor_state']['sensed'][name][field]-=20_000_000
    model=MomentWeakSubspaceIntegrator()
    with pytest.raises(SensorContractError): model.observe(p,original.state(p))
    assert model.failed
