"""Causal visual course and small supervised action-response baselines.

No native state or terrain/friction label. Window scatter is not uncertainty.
"""
from collections import deque
import math
import numpy as np

from lewm.causal_sensor_state import SensorContractError, _ns
from lewm.bounded_visual_servo_development import wrapped


class VisualCourseWindow:
    """Six 10Hz poses, least-squares planar velocity over the past 0.5 seconds.

    Course is a window measurement, not instantaneous velocity. Transporting its
    offset from mean body yaw to current yaw assumes slowly varying slip.
    """
    def __init__(self): self.rows=deque(maxlen=6)

    def observe(self, *, measured_ns, position_initial_xy_m, yaw_rad):
        stamp=_ns(measured_ns,'course measurement'); p=np.asarray(position_initial_xy_m,float)
        if p.shape!=(2,) or not np.isfinite(p).all() or not np.isfinite(yaw_rad):
            raise SensorContractError('finite planar visual pose required')
        if self.rows and stamp-self.rows[-1][0]!=100_000_000:
            raise SensorContractError('complete exact 10Hz course window required')
        self.rows.append((stamp,p.copy(),float(yaw_rad)))
        base=dict(measured_ns=stamp,samples=len(self.rows),course_error_bound_rad=None,
                  physical_velocity_error_bound_m_s=None,window_residual_is_not_covariance=True)
        if len(self.rows)<6:
            return base|dict(status='COURSE_WINDOW_INCOMPLETE',velocity_initial_xy_m_s=None,
                             course_rad=None,body_course_offset_rad=None,current_course_hypothesis_rad=None)
        times=np.array([(r[0]-stamp)*1e-9 for r in self.rows]); t=times-times.mean()
        positions=np.array([r[1] for r in self.rows]); centred=positions-positions.mean(0)
        v=(t[:,None]*centred).sum(0)/(t@t); residual=centred-t[:,None]*v
        angles=np.unwrap([r[2] for r in self.rows]); rate=float(t@(angles-angles.mean())/(t@t))
        speed=float(np.linalg.norm(v)); course=math.atan2(v[1],v[0]) if speed>=.012 else None
        offset=None if course is None else wrapped(course-float(angles.mean()))
        return base|dict(status='COURSE_AVAILABLE' if course is not None else 'COURSE_LOW_SPEED',
            start_ns=self.rows[0][0],velocity_initial_xy_m_s=v.tolist(),speed_m_s=speed,
            mean_body_yaw_rad=float(angles.mean()),mean_yaw_rate_rad_s=rate,
            course_rad=course,body_course_offset_rad=offset,
            current_course_hypothesis_rad=None if offset is None else wrapped(yaw_rad+offset),
            rms_position_residual_m=float(np.sqrt(np.mean(np.sum(residual**2,axis=1)))),
            slowly_varying_body_course_offset_assumed=True)


def action_features(command,course,current_yaw):
    """Deployable regression features: proposed twist and past measured motion."""
    u=np.asarray(command,float)
    if u.shape!=(3,) or not np.isfinite(u).all() or u[1]!=0:
        raise ValueError('finite forward/yaw command required')
    v=np.asarray(course['velocity_initial_xy_m_s'],float)
    if v.shape!=(2,) or not np.isfinite(v).all(): raise ValueError('complete past visual course required')
    c,s=math.cos(current_yaw),math.sin(current_yaw)
    body=np.array([[c,s],[-s,c]])@v
    return np.array([1.,u[0],u[2],u[0]*u[2],body[0],body[1],course['mean_yaw_rate_rad_s']])


def fit_response(features,targets):
    X,Y=np.asarray(features,float),np.asarray(targets,float)
    if X.ndim!=2 or X.shape[1]!=7 or Y.shape!=(len(X),3) or len(X)<8 or not np.isfinite([X.sum(),Y.sum()] ).all():
        raise ValueError('finite aligned response fitting rows required')
    if not np.isfinite(X).all() or not np.isfinite(Y).all(): raise ValueError('finite response data required')
    scale=np.sqrt(np.mean(X**2,axis=0)); scale=np.where(scale>1e-12,scale,1.)
    A=X/scale
    penalty=np.eye(7)*.01; penalty[0,0]=0.
    coefficients=np.linalg.solve(A.T@A+penalty,A.T@Y)/scale[:,None]
    singular=np.linalg.svd(A,compute_uv=False)
    return dict(coefficients=coefficients.tolist(),feature_rms_scale=scale.tolist(),
        standardized_rank=int(np.linalg.matrix_rank(A)),standardized_singular_values=singular.tolist(),
        action_design_rank=int(np.linalg.matrix_rank(X[:,:4])),fitting_rows=len(X),
        ridge_penalty=.01,causal_action_effect_identified=False,physical_error_bound=None)
