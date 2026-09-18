"""Experimental gyro-conditioned initial proposals, before floor refinement.

Use the already declared gyro rotation estimator throughout single/pooled
camera proposal fitting. Keep match populations, consensus/conditioning gates,
floor refinement, final gyro refit and temporal admission unchanged.
"""
from lewm.causal_sensor_state import SensorContractError
from lewm.eligible_floor_registration_development import bind
from lewm.development_support_tracker_development import use
from lewm.conditioned_support_tracker_development import (
    CONDITIONED_RULES, _Joint, _Dual, _Direct, _Chained)
from lewm.full_consensus_early_exit_development import register as full_register
from lewm.joint_rgbd_rigid_pose_development import inliers
from lewm.joint_camera_registration_development import register_views as original_views
from lewm.joint_camera_anchor_tracker_development import _JointCameras
from lewm.consecutive_retained_camera_pair_development import (
    ConsecutiveRetainedCameraPairPose, ConsecutiveRetainedCameraPairMotion)


def gyro_core(*args, **kwargs):
    if kwargs.get('mode') not in ('joint', 'gyro'):
        raise SensorContractError('declared original fitting mode required')
    # register_views privately replaces `inliers` with its calibrated camera
    # projection checker; preserve that binding in the numerical fit.
    fit = bind(full_register, RULES=CONDITIONED_RULES, inliers=inliers)
    R, t, mask, receipt = fit(*args, **(kwargs | dict(mode='gyro')))
    return R, t, mask, receipt | dict(gyro_conditioned_initial_proposals=True,
        initial_unconstrained_joint_fit_required=False,
        absolute_geometric_thresholds_changed=False)


def register(*args, **kwargs):
    R, t, mask, receipt = gyro_core(*args, **kwargs)
    if 2*int(mask.sum()) <= len(mask):
        raise SensorContractError('strict inlier majority required')
    return R, t, mask, receipt | dict(image_bin_count_gate_used=False,
        strict_inlier_majority_required=True, original_measured_3d_conditioning_required=True)


register_views = bind(original_views, original_register=gyro_core)


class _InitialJoint(_Joint):
    _candidate = use(_Joint._candidate, register=register)


class _InitialDual(_Dual, _InitialJoint):
    _candidate = use(_Dual._candidate, register=register)


class _InitialDirect(_Direct, _InitialDual):
    _candidate = use(_Direct._candidate, register=register)


class _InitialChained(_Chained, _InitialDirect):
    _candidate = use(_Chained._candidate, register=register)


class _InitialViews(_JointCameras, _InitialChained):
    _candidate = use(_JointCameras._candidate, register_views=register_views)


class GyroInitialCameraConsensusPose(ConsecutiveRetainedCameraPairPose, _InitialViews):
    pass


class GyroInitialCameraConsensusMotion(ConsecutiveRetainedCameraPairMotion):
    def __init__(self, *, identity=(0, 0, 0), activation_frame=0):
        super().__init__(identity=identity, activation_frame=activation_frame)
        self.model = GyroInitialCameraConsensusPose(activation_frame=activation_frame)

    def snapshot(self, *, now_ns):
        return super().snapshot(now_ns=now_ns) | dict(
            gyro_conditioned_initial_camera_consensus_enabled=True,
            initial_single_and_pooled_camera_fitting='gyro',
            initial_unconstrained_joint_fit_required=False)
