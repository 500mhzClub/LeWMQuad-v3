"""Small supervised SENSOR-target response model, not a navigation policy/JEPA.

Fixed features and ridge penalty; no native truth is a training or runtime input.
Predicted state is propagated for every planned action, without future observations.
"""
from copy import deepcopy
import hashlib
import json

import numpy as np

from lewm.articulated_trajectory_evidence_development import command_baseline_trajectory
from lewm.causal_sensor_state import SensorContractError
from lewm.relative_gyro_turn_development import rotation_increment

SCHEMA='sensor_action_response_ridge.v1'
RIDGE=.01
Q_REFERENCE=np.repeat([0.,.8,-1.5],4)


def features(q,dq,velocity,gyro,prior,applied):
    q,dq,velocity,gyro,prior,applied=[np.asarray(x,float) for x in (q,dq,velocity,gyro,prior,applied)]
    if any(x.shape!=shape or not np.isfinite(x).all() for x,shape in
           zip((q,dq,velocity,gyro,prior,applied),((12,),(12,),(3,),(3,),(3,),(3,)),strict=True)):
        raise SensorContractError('finite ordered response state and commands required')
    controls=np.r_[prior[[0,2]]/[.3,.5],applied[[0,2]]/[.3,.5]]
    body=np.r_[1.,velocity/.3,gyro/.5,controls]
    joints=np.column_stack((np.ones(12),(q-Q_REFERENCE)/.5,dq/5.,np.tile(controls,(12,1))))
    return body,joints


def ridge_fit(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if (x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or not len(x)
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise SensorContractError('finite aligned response training rows required')
    penalty=RIDGE*np.eye(x.shape[1]); penalty[0,0]=0.
    return np.linalg.solve(x.T@x+penalty,x.T@y)


def fit_response(rows):
    if not rows: raise SensorContractError('nonempty A-only sensor training rows required')
    x=np.asarray([r['body_features'] for r in rows]); y=np.asarray([r['body_target'] for r in rows])
    jx=np.asarray([r['joint_features'] for r in rows]); jy=np.asarray([r['joint_target'] for r in rows])
    if x.shape!=(len(rows),11) or y.shape!=(len(rows),12) or jx.shape!=(len(rows),12,7) or jy.shape!=(len(rows),12,2):
        raise SensorContractError('fixed body and per-joint feature/target layout required')
    return dict(schema=SCHEMA,ridge=RIDGE,training_rows=len(rows),
        body_weights=ridge_fit(x,y).tolist(),joint_weights=np.stack([ridge_fit(jx[:,j],jy[:,j]) for j in range(12)]).tolist(),
        feature_normalization='fixed_v0.3_w0.5_q0.5_dq5_command0.3_0.5',
        target_normalization='delta_position0.03_rotation0.05_next_v0.3_w0.5_delta_q0.5_next_dq5',
        training_targets='deployment_valid_future_depth_gyro_joint_observations',
        native_training_inputs=False,independent_validation_complete=False)


def validate_model(model):
    expected={'schema','ridge','training_rows','body_weights','joint_weights','feature_normalization',
              'target_normalization','training_targets','native_training_inputs','independent_validation_complete'}
    if (not isinstance(model,dict) or set(model)!=expected or model['schema']!=SCHEMA or model['ridge']!=RIDGE
            or type(model['training_rows']) is not int or model['training_rows']<=0
            or model['feature_normalization']!='fixed_v0.3_w0.5_q0.5_dq5_command0.3_0.5'
            or model['target_normalization']!='delta_position0.03_rotation0.05_next_v0.3_w0.5_delta_q0.5_next_dq5'
            or model['training_targets']!='deployment_valid_future_depth_gyro_joint_observations'
            or model['native_training_inputs'] is not False or model['independent_validation_complete'] is not False):
        raise SensorContractError('exact unvalidated sensor-response model schema required')
    for name,shape in (('body_weights',(11,12)),('joint_weights',(12,7,2))):
        array=np.asarray(model[name],float)
        if array.shape!=shape or not np.isfinite(array).all(): raise SensorContractError('finite fixed response coefficients required')


def model_identity(model):
    validate_model(model)
    return hashlib.sha256(json.dumps(model,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def policy_state(owner,policy,*,now_ns):
    # Reuse the strict current packet/owner identity and joint/command checks.
    baseline=command_baseline_trajectory(owner,policy,[[0.,0.,0.]],now_ns=now_ns)
    gyro=policy['sensor_state']['sensed']['gyro']
    if gyro['measured_ns'][-1]!=now_ns or not np.asarray(gyro['valid']).all():
        raise SensorContractError('fresh complete body gyro required')
    consumed=owner._memory._rays.integrator.gyro_history
    if any(not np.array_equal(gyro[k],consumed[k]) for k in ('values','valid','measured_ns','available_ns')):
        raise SensorContractError('gyro history differs from the current state owner')
    values=np.asarray(policy['sensor_state']['sensed']['joints']['values'][-1],float)
    velocity=owner._memory._rays.rotation.T@np.asarray(owner._memory._rays.fusion['velocity_initial_body_m_s'])
    state=dict(q=values[:12].copy(),dq=values[12:].copy(),velocity=velocity,
        gyro=np.asarray(gyro['values'][-1],float).copy(),
        prior=np.asarray(policy['sensor_state']['control']['applied_command']['values'][-1],float).copy())
    return state,baseline


def predict_response(owner,policy,requested_commands,model,*,now_ns,model_sha256):
    validate_model(model)
    if not isinstance(model_sha256,str) or len(model_sha256)!=64 or any(c not in '0123456789abcdef' for c in model_sha256):
        raise SensorContractError('frozen model content identity required')
    if model_identity(model)!=model_sha256: raise SensorContractError('response coefficients differ from frozen canonical identity')
    baseline=command_baseline_trajectory(owner,policy,requested_commands,now_ns=now_ns)
    state,_=policy_state(owner,policy,now_ns=now_ns)
    q,dq,v,w,prior=[state[k] for k in ('q','dq','velocity','gyro','prior')]
    p=np.zeros(3); R=np.eye(3); positions=[p.tolist()]; rotations=[R.tolist()]; joints=[q.tolist()]; predicted_states=[]
    body_weights=np.asarray(model['body_weights']); joint_weights=np.asarray(model['joint_weights'])
    initial_features=[]
    for applied in np.asarray(baseline['expected_applied_commands']):
        bx,jx=features(q,dq,v,w,prior,applied)
        if not initial_features: initial_features=[bx.tolist(),jx.tolist()]
        by=bx@body_weights; jy=np.einsum('jk,jko->jo',jx,joint_weights)
        p=p+R@(by[:3]*.03); R=R@rotation_increment(by[3:6]*.05)
        q=q+jy[:,0]*.5; dq=jy[:,1]*5.; v=by[6:9]*.3; w=by[9:12]*.5; prior=applied.copy()
        if not all(np.isfinite(a).all() for a in (p,R,q,dq,v,w)): raise SensorContractError('nonfinite response rollout')
        positions.append(p.tolist()); rotations.append(R.tolist()); joints.append(q.tolist())
        predicted_states.append(dict(joint_velocity_rad_s=dq.tolist(),body_velocity_m_s=v.tolist(),body_gyro_rad_s=w.tolist()))
    return baseline | dict(model_id=SCHEMA,model_sha256=model_sha256,positions_current_body_m=positions,
        rotations_current_body=rotations,joints_rad=joints,predicted_states=predicted_states,
        initial_features_sha256=hashlib.sha256(json.dumps(initial_features,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
        learned_prediction=True,training_targets='future_deployment_valid_sensors',execution_error_validated=False,
        stopping_model_validated=False,navigation_action_permitted=False)


def position_persistence(baseline):
    result=deepcopy(baseline); result['model_id']='ideal_command_se2_joint_position_persistence.v1'
    result['joints_rad']=[deepcopy(result['joints_rad'][0]) for _ in result['offsets_ns']]
    return result
