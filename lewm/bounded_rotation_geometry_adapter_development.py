"""Explicit numerical rotation correction with per-shape displacement allowance.

Does not alter estimator history or physical uncertainty. The finite input
matrix and returned proper matrix define different point maps; their difference
is bounded, not silently discarded. Unknown sensor/body errors stay unknown.
"""
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.joint_rgbd_rigid_pose_development import proper


def rotation_for_geometry(rotation):
    R=proper(rotation).copy()
    U,_,Vt=np.linalg.svd(R)
    Q=U@Vt
    if (np.linalg.det(Q)<=0 or not np.allclose(Q.T@Q,np.eye(3),rtol=0,atol=1e-12)
            or abs(np.linalg.det(Q)-1)>1e-12):
        raise SensorContractError('strict proper projected rotation unavailable')
    # Frobenius norm dominates spectral norm: ||(R-Q)x|| <= ||R-Q||F ||x||.
    # Numerical guard concerns only these finite matrices, not sensor errors.
    guard=128*np.finfo(float).eps*(1+np.linalg.norm(R,ord='fro')+np.linalg.norm(Q,ord='fro'))
    upper=float(np.linalg.norm(R-Q,ord='fro')+guard)
    return Q,dict(matrix_difference_frobenius_upper=upper,numerical_guard=guard,
        estimator_rotation_unchanged=True,physical_uncertainty_calibrated=False)


def shape_corrections(shapes, matrix_difference_upper):
    if not np.isfinite(matrix_difference_upper) or matrix_difference_upper<0:
        raise SensorContractError('finite nonnegative matrix-difference bound required')
    corrections={}
    for s in shapes:
        lo,hi=np.asarray(s['lower'],float),np.asarray(s['upper'],float)
        if (lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite([lo,hi]).all()
                or np.any(lo>hi) or s['shape_id'] in corrections):
            raise SensorContractError('unique finite ordered body-shape boxes required')
        lever=float(np.linalg.norm(np.maximum(abs(lo),abs(hi))))
        corrections[s['shape_id']]=float(np.nextafter(matrix_difference_upper*lever,np.inf))
    if not corrections: raise SensorContractError('nonempty body-shape roster required')
    return corrections


def adapt_geometry_query(geometry,joints,state,physical_errors):
    if state is None: raise SensorContractError('accepted pose required')
    R,details=rotation_for_geometry(state['rotation_initial_body_from_current_body'])
    shapes=geometry.supports(joints,np.eye(3))['shapes']
    correction=shape_corrections(shapes,details['matrix_difference_frobenius_upper'])
    if physical_errors is None:
        raise SensorContractError('unknown physical errors cannot become zero during numerical adaptation')
    if (not isinstance(physical_errors,dict) or set(physical_errors)!=set(correction)
            or any(np.asarray(v).shape!=() or not np.isfinite(v) or v<0 for v in physical_errors.values())):
        raise SensorContractError('explicit finite per-shape physical error hypotheses required')
    total={k:float(np.nextafter(physical_errors[k]+correction[k],np.inf)) for k in correction}
    return dict(rotation_observation_from_body=R,
        translation_observation_from_body=list(state['position_initial_body_m']),point_error_by_shape=total),details|dict(
        numerical_shape_correction_m=correction,physical_error_hypotheses=dict(physical_errors),
        navigation_qualified=False,future_gait_qualified=False)
