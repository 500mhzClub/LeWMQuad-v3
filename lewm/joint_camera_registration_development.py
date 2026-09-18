"""Fit synchronized within-camera correspondences in one calibrated body frame."""
import numpy as np
from lewm.auxiliary_reference_pose_adapter_development import body_from_reference
from lewm.rgbd_correspondence_motion_development import project
from lewm.joint_rgbd_rigid_pose_development import inliers, SensorContractError
from lewm.full_consensus_early_exit_development import register as original_register
from lewm.conditioned_support_tracker_development import CONDITIONED_RULES, bind


def project_views(points, split):
    A, offset = body_from_reference()
    p, valid_p = project(points[:split])
    q, valid_q = project((points[split:]-offset)@A)
    return np.concatenate((p, q)), np.concatenate((valid_p, valid_q))


def register_views(primary, auxiliary, *, gyro_rotation, frame):
    A, offset = body_from_reference()
    pa, pb, pu, pv = primary
    aa, ab, au, av = auxiliary
    a = np.concatenate((pa, aa@A.T+offset))
    b = np.concatenate((pb, ab@A.T+offset))
    ua, ub = np.concatenate((pu, au)), np.concatenate((pv, av))
    split = len(pa)

    def camera_project(points):
        if len(points) != len(a):
            raise SensorContractError('camera identities require full correspondence array')
        return project_views(points, split)

    measured_inliers = bind(inliers, project=camera_project)
    fit = bind(original_register, RULES=CONDITIONED_RULES, inliers=measured_inliers)
    R, t, mask, receipt = fit(a, b, ua, ub, gyro_rotation=gyro_rotation, mode='joint', frame=frame)
    if 2*int(mask.sum()) <= len(mask):
        raise SensorContractError('strict inlier majority required')
    counts = [int(mask[:split].sum()), int(mask[split:].sum())]
    if min(counts) < 3:
        raise SensorContractError('joint camera fit requires at least three inliers from each view')
    return R, t, mask, receipt | dict(camera_inliers=counts,
        camera_matches=[len(pa), len(aa)], calibrated_body_frame_fit=True,
        camera_specific_reprojection=True), (a, b, ua, ub)
