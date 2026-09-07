from copy import deepcopy

import numpy as np
import pytest

from lewm.articulated_trajectory_evidence_development import command_baseline_trajectory, evaluate_trajectory
from lewm.causal_sensor_state import SensorContractError
from lewm.tests.test_continuous_startup_handoff_development import frames
from lewm.tests.test_observed_setup_configuration_development import ready


def example(commands=((.3,0.,.5),(0.,0.,0.))):
    owner, now = ready(); policy = list(frames(8))[-1][0]
    return owner, now, policy, command_baseline_trajectory(owner,policy,commands,now_ns=now)


def test_action_sequence_slew_gravity_tangent_and_nonlearned_provenance():
    owner, now, policy, prediction = example()
    np.testing.assert_allclose(prediction['expected_applied_commands'],[[.25,0.,.35],[0.,0.,0.]])
    assert prediction['offsets_ns']==[0,100_000_000,200_000_000]
    np.testing.assert_allclose(np.asarray(prediction['positions_current_body_m'])@prediction['up_current_body'],0.,atol=1e-12)
    assert len(prediction['joint_command_history_sha256'])==64
    assert not prediction['learned_prediction'] and not prediction['execution_error_validated']
    assert not prediction['stopping_model_validated'] and not prediction['navigation_action_permitted']
    stop=command_baseline_trajectory(owner,policy,[[0.,0.,0.],[0.,0.,0.]],now_ns=now)
    assert prediction['positions_current_body_m']!=stop['positions_current_body_m']


def test_joints_are_time_indexed_measured_velocity_baseline_not_fixed_posture():
    owner, now, policy, _ = example()
    policy=deepcopy(policy); policy['sensor_state']['sensed']['joints']['values'][-1,12:]=.1
    prediction=command_baseline_trajectory(owner,policy,[[0.,0.,0.]]*3,now_ns=now)
    np.testing.assert_allclose(np.asarray(prediction['joints_rad'])[-1]-owner._memory._joints,.03,atol=1e-8)


def test_nearest_endpoint_enclosure_passes_all_nodes_and_full_horizon(monkeypatch):
    import lewm.articulated_trajectory_evidence_development as module
    owner, now, _, prediction = example(); calls=[]
    def capture(*args, **kwargs):
        calls.append((args,kwargs)); return dict(all_primitives_conditionally_nonfloor_clear=True)
    monkeypatch.setattr(module,'query_factored_configuration',capture)
    result=evaluate_trajectory(owner,prediction,point_errors_m=[.01,.02,.03],
        physical_point_speed_bounds_m_s=[.2,.6],now_ns=now)
    np.testing.assert_allclose(result['intersample_expanded_point_errors_m'],[.02,.05,.06])
    assert len(calls)==3 and all(c[1]['through_ns']==now+200_000_000 for c in calls)
    assert result['all_nodes_conditionally_nonfloor_clear']
    assert not result['continuous_swept_volume_established'] and not result['navigation_action_permitted']


def test_real_trajectory_queries_remain_separate_from_support_and_execution():
    owner, now, _, prediction=example(commands=((.08,0.,0.),(0.,0.,0.)))
    result=evaluate_trajectory(owner,prediction,point_errors_m=[0.,.01,.02],
        physical_point_speed_bounds_m_s=[.5,.5],now_ns=now)
    assert len(result['configuration_queries'])==3
    assert not result['motion_and_error_assumptions_validated'] and not result['ground_support_permission']


@pytest.mark.parametrize('fault', ['stale','identity','privileged','missing_joint','anchor_joint','command_domain'])
def test_baseline_uses_only_fresh_bound_sensor_inputs(fault):
    owner, now, policy, _=example(); commands=[[.1,0.,0.]]
    if fault=='stale': now-=1
    if fault=='identity': policy['sensor_state']['identity']=(0,0,9)
    if fault=='privileged': policy['native_pose']=[0.,0.,0.]
    if fault=='missing_joint':
        source=policy['sensor_state']['sensed']['joints']; source['valid'][-1,0]=False; source['values'][-1,0]=0.
    if fault=='anchor_joint': policy['sensor_state']['sensed']['joints']['values'][-1,0]+=.01
    if fault=='command_domain': commands=[[.1,.01,0.]]
    with pytest.raises(SensorContractError): command_baseline_trajectory(owner,policy,commands,now_ns=now)


@pytest.mark.parametrize('fault', ['nan_error','negative_speed','duplicate_time','bad_rotation','wrong_anchor','wrong_joints','bad_shapes'])
def test_trajectory_requires_explicit_valid_anchored_motion_assumptions(fault):
    owner, now, _, prediction=example(); errors=[0.,.01,.02]; speeds=[.5,.5]
    if fault=='nan_error': errors[1]=float('nan')
    if fault=='negative_speed': speeds[1]=-.1
    if fault=='duplicate_time': prediction['offsets_ns'][1]=0
    if fault=='bad_rotation': prediction['rotations_current_body'][1][0][0]=2.
    if fault=='wrong_anchor': prediction['anchor_ns']-=1
    if fault=='wrong_joints': prediction['joints_rad'][0][0]+=.01
    if fault=='bad_shapes': prediction['positions_current_body_m']=[[0.,0.,0.]]
    with pytest.raises(SensorContractError):
        evaluate_trajectory(owner,prediction,point_errors_m=errors,physical_point_speed_bounds_m_s=speeds,now_ns=now)
