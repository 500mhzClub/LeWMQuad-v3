"""Rigid coordinate adapter for existing optical lifting/registration helpers.

The helpers lift through the primary fixed extrinsic. For auxiliary images
this defines a reference coordinate frame, not the robot body frame. Convert
gyro rotations into that frame before fitting and convert fitted poses back.
No observed/native robot pose is used to define this fixed calibration.
"""
import numpy as np
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.joint_rgbd_rigid_pose_development import proper


def body_from_reference():
    primary=np.asarray(BODY_FROM_OPTICAL,float);auxiliary=body_from_optical()
    R=auxiliary[:3,:3]@primary[:3,:3].T
    t=auxiliary[:3,3]-R@primary[:3,3]
    return R,t


def gyro_in_reference(rotation_previous_body_from_current_body):
    R,_=body_from_reference()
    return proper(R.T@proper(rotation_previous_body_from_current_body)@R)


def pose_in_body(rotation_previous_reference_from_current_reference,translation_previous_reference_m):
    A,b=body_from_reference();Q=proper(rotation_previous_reference_from_current_reference)
    t=np.asarray(translation_previous_reference_m,float)
    if t.shape!=(3,) or not np.isfinite(t).all():raise ValueError('finite reference-frame translation required')
    rotation=proper(A@Q@A.T)
    translation=b+A@t-rotation@b
    return rotation,translation
