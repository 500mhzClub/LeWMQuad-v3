from copy import deepcopy

import numpy as np
import pytest

from lewm.action_response_model_development import (
    features,fit_response,validate_model,model_identity,predict_response,position_persistence)
from lewm.articulated_trajectory_evidence_development import command_baseline_trajectory
from lewm.causal_sensor_state import SensorContractError
from lewm.tests.test_continuous_startup_handoff_development import frames
from lewm.tests.test_observed_setup_configuration_development import ready


def fitted():
    rng=np.random.default_rng(2026090604); rows=[]
    for _ in range(60):
        bx=np.r_[1.,rng.normal(size=10)];jx=np.column_stack((np.ones(12),rng.normal(size=(12,6))))
        by=np.zeros(12);by[0]=bx[-2];by[6]=bx[-2]
        jy=np.zeros((12,2));jy[:,0]=.01*jx[:,2];jy[:,1]=.2*jx[:,2]
        rows.append(dict(body_features=bx,body_target=by,joint_features=jx,joint_target=jy))
    return fit_response(rows)


def test_fixed_sensor_model_is_deterministic_finite_and_content_bound():
    a,b=fitted(),fitted();validate_model(a)
    assert a==b and model_identity(a)==model_identity(b)
    assert np.asarray(a['body_weights']).shape==(11,12)
    assert not a['native_training_inputs'] and not a['independent_validation_complete']


def test_features_use_current_state_and_next_action_in_declared_order():
    bx,jx=features(np.repeat([0.,.8,-1.5],4),np.zeros(12),[.3,0,0],[0,0,.5],[.15,0,.25],[.3,0,-.5])
    np.testing.assert_array_equal(bx,[1,1,0,0,0,0,1,.5,.5,1,-1])
    np.testing.assert_array_equal(jx[0],[1,0,0,.5,.5,1,-1])


def test_multistep_response_is_action_conditioned_and_does_not_use_future_packets():
    owner,now=ready();policy=list(frames(8))[-1][0];model=fitted();identity=model_identity(model)
    a=predict_response(owner,policy,[[.1,0,0]]*4,model,now_ns=now,model_sha256=identity)
    b=predict_response(owner,policy,[[0.,0,0]]*4,model,now_ns=now,model_sha256=identity)
    assert len(a['predicted_states'])==4 and a['positions_current_body_m']!=b['positions_current_body_m']
    assert a['learned_prediction'] and not a['navigation_action_permitted'] and not a['execution_error_validated']
    assert a['joints_rad'][0]==owner._memory._joints.tolist()
    model['body_weights'][0][0]+=.1
    with pytest.raises(SensorContractError): predict_response(owner,policy,[[0.,0,0]],model,now_ns=now,model_sha256=identity)


def test_position_control_preserves_body_commands_and_original_prediction():
    owner,now=ready();policy=list(frames(8))[-1][0]
    a=command_baseline_trajectory(owner,policy,[[.1,0,0]]*4,now_ns=now); before=deepcopy(a);b=position_persistence(a)
    assert a==before and b['positions_current_body_m']==a['positions_current_body_m']
    assert all(q==a['joints_rad'][0] for q in b['joints_rad'])


@pytest.mark.parametrize('fault',['extra','nonfinite','shape','ridge','native','validation'])
def test_model_schema_rejects_unbound_or_altered_fit_contract(fault):
    model=fitted()
    if fault=='extra':model['hidden_state']=[]
    if fault=='nonfinite':model['body_weights'][0][0]=float('nan')
    if fault=='shape':model['joint_weights']=[]
    if fault=='ridge':model['ridge']=.02
    if fault=='native':model['native_training_inputs']=True
    if fault=='validation':model['independent_validation_complete']=True
    with pytest.raises(SensorContractError):validate_model(model)
