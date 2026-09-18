"""Use integrated gyro only to initialize image search, then fit measured pairs.

Both directions initialize from their own measured depth. This supplies no
accepted pose and does not change the existing pixel, photometric or depth
checks. Image associations now depend on gyro; it is not an independent-only
consistency monitor for these pairs.
"""
from types import SimpleNamespace

import cv2
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.direct_corner_flow_association_development import tracked_points as original
from lewm.eligible_floor_registration_development import bind
from lewm.joint_rgbd_rigid_pose_development import proper
from lewm.rgbd_correspondence_motion_development import lift, project


def initial_pixels(depth, pixels, rotation, translation):
    points, valid = lift(depth, pixels[:, 0])
    projected, front = project((points-translation)@rotation)
    usable = valid & front & np.isfinite(projected).all(1)
    usable &= (projected[:, 0] >= 0) & (projected[:, 0] < 640)
    usable &= (projected[:, 1] >= 0) & (projected[:, 1] < 480)
    seeded = pixels.copy()
    seeded[usable, 0] = projected[usable]
    return seeded, int(usable.sum())


def tracked_points(reference, current, *, rotation, translation):
    R = proper(rotation); t = np.asarray(translation, float)
    if t.shape != (3,) or not np.isfinite(t).all():
        raise SensorContractError('finite camera-origin translation hypothesis required')
    seeds = []
    def flow(left, right, pixels, unused, **options):
        if left is reference.gray and right is current.gray:
            initial, count = initial_pixels(reference.depth, pixels, R, t)
            direction = 'forward'
        elif left is current.gray and right is reference.gray:
            initial, count = initial_pixels(current.depth, pixels, R.T, -R.T@t)
            direction = 'reverse'
        else: raise SensorContractError('original measured image pair required')
        seeds.append(dict(direction=direction, points=len(pixels), gyro_projected_points=count))
        return cv2.calcOpticalFlowPyrLK(left, right, pixels, initial,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **options)
    api = SimpleNamespace(calcOpticalFlowPyrLK=flow,
        TERM_CRITERIA_COUNT=cv2.TERM_CRITERIA_COUNT, TERM_CRITERIA_EPS=cv2.TERM_CRITERIA_EPS)
    values, receipt = bind(original, cv2=api)(reference, current)
    receipt.update(association='gyro_seeded_direct_corner_lk_photometric_v1',
        initial_flow='gyro_rotation_and_zero_body_translation_hypothesis',
        seed_counts=seeds, gyro_role='association_initializer_and_consistency_monitor',
        both_directions_use_own_measured_depth=True,
        original_pixel_photometric_depth_gates_unchanged=True,
        gyro_supplies_accepted_pose=False)
    return values, receipt
