"""Explicit joint-pose error propagation, not a covariance calibration method.

No covariance is inferred from the existing scalar pose-error proxies. This
primitive requires a complete joint error model, including cross-correlation.
It is not connected to clearance approval until that model is established.
"""
import numpy as np

from lewm.causal_sensor_state import SensorContractError


def skew(vector):
    x,y,z=vector
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def relative_point_moments(points,current_position,current_rotation,stored_position,stored_rotation,joint_covariance):
    """First-order propagation of correlated position and left-rotation errors.

    Error order is (current position, current rotation, stored position,
    stored rotation), each xyz in the shared reference; units are m/rad.
    A shared rigid reference error cancels, including its position/rotation
    cross terms. Independent and anticorrelated errors do not cancel.
    """
    points=np.asarray(points,dtype=float)
    pc,rc,pf,rf,cov=[np.asarray(x,dtype=float) for x in
        (current_position,current_rotation,stored_position,stored_rotation,joint_covariance)]
    if (points.ndim!=2 or points.shape[1:]!=(3,) or pc.shape!=(3,) or pf.shape!=(3,)
            or rc.shape!=(3,3) or rf.shape!=(3,3) or cov.shape!=(12,12)
            or not all(np.isfinite(x).all() for x in (points,pc,rc,pf,rf,cov))
            or not np.allclose(cov,cov.T,atol=1e-12,rtol=0)):
        raise SensorContractError('finite poses and complete symmetric joint covariance required')
    for rotation in (rc,rf):
        if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-8,rtol=0) or abs(np.linalg.det(rotation)-1)>1e-8:
            raise SensorContractError('proper pose rotations required')
    if np.linalg.eigvalsh(cov).min()<-1e-12:
        raise SensorContractError('positive semidefinite joint covariance required')
    reference=points@rc.T
    mean=(reference+pc-pf)@rf
    jacobian=np.empty((len(points),3,12))
    for i,p in enumerate(reference):
        jacobian[i]=np.concatenate((rf.T,-rf.T@skew(p),-rf.T,rf.T@skew(p+pc-pf)),axis=1)
    propagated=np.einsum('nij,jk,nlk->nil',jacobian,cov,jacobian)
    return {'point_mean_stored_body_m':mean,'point_covariance_m2':propagated,'jacobian':jacobian,
        'first_order_only':True,'joint_input_calibration_established':False,
        'navigation_qualified':False,'hardware_qualified':False}


def shared_rigid_error_jacobian(current_position,stored_position):
    """Joint-pose effect of common translation/rotation of the reference."""
    pc,pf=[np.asarray(x,dtype=float) for x in (current_position,stored_position)]
    if pc.shape!=(3,) or pf.shape!=(3,) or not np.isfinite(pc).all() or not np.isfinite(pf).all():
        raise SensorContractError('finite shared-reference positions required')
    result=np.zeros((12,6))
    result[:3,:3]=np.eye(3); result[:3,3:]=-skew(pc); result[3:6,3:]=np.eye(3)
    result[6:9,:3]=np.eye(3); result[6:9,3:]=-skew(pf); result[9:12,3:]=np.eye(3)
    return result
