"""Development-only gyro-seeded image tracks with unchanged rigid-pose gates.

Rotation seeds flow; it is not a correspondence measurement. This replaces
mutual descriptor association and descriptor-seed proximity, not the subsequent
rigid consensus/reprojection/grid/increment checks. Repeated texture and common
sensor errors can defeat these conditional checks. No calibrated uncertainty,
command-based translation prior, native input, or permission to navigate.
"""
from copy import deepcopy

import cv2
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.joint_rgbd_rigid_pose_development import proper, register, angle, RIGID_RULES
from lewm.multi_reference_rgbd_pose_development import MultiReferenceRGBDPose, MultiReferenceVisualLedMotion
from lewm.rgbd_correspondence_motion_development import RULES, lift, project


def inside(uv):
    return np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < 639) & (uv[:, 1] >= 0) & (uv[:, 1] < 479)


def validate_frame(frame):
    gray = frame.gray
    z, valid = frame.depth['depth_m'], frame.depth['valid']
    if (not isinstance(gray, np.ndarray) or gray.shape != (480, 640) or gray.dtype != np.uint8
            or not isinstance(z, np.ndarray) or z.shape != gray.shape or z.dtype != np.float32
            or not isinstance(valid, np.ndarray) or valid.shape != z.shape or valid.dtype != bool
            or not np.isfinite(z).all() or np.any(z[~valid] != 0.)
            or np.any(z[valid] < .2) or np.any(z[valid] > 5.)):
        raise SensorContractError('fixed image geometry and finite metric depth/mask required')


def matched_points(reference, current, rotation):
    """Return measured point pairs and stage counts; never fit a translation seed.

    R maps current-body vectors to reference-body vectors. With a zero body
    translation *initial guess*, a reference body point a projects as a @ R
    in the current body. project() includes the calibrated camera lever arm.
    The actual displacement is estimated later from measured RGB-D pairs.
    """
    R = proper(rotation)
    validate_frame(reference); validate_frame(current)
    counts = dict(reference_keypoints=len(reference.keypoints), unique_reference_points=0,
        reference_depth_survivors=0, projected_seeds=0, forward_flow_survivors=0,
        bidirectional_flow_survivors=0, unique_current_points=0, paired_depth_survivors=0,
        descriptor_association_used=False, command_translation_prior_used=False,
        gyro_seed_is_measured_correspondence=False, uncertainty_calibrated=False)
    empty = (np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2)), np.empty((0, 2)))
    points = []; seen = set()
    for keypoint in reference.keypoints:
        p = np.asarray(keypoint.pt, float)
        if p.shape != (2,) or not np.isfinite(p).all():
            raise SensorContractError('finite reference feature coordinates required')
        if not inside(p[None])[0]: continue
        cell = tuple(np.rint(p * 2).astype(int))
        if cell in seen: continue
        points.append(p); seen.add(cell)
        if len(points) == RULES['sift_features']: break
    counts['unique_reference_points'] = len(points)
    if not points: return (*empty, counts)
    uv = np.asarray(points); a, valid = lift(reference.depth, uv)
    uv, a = uv[valid], a[valid]; counts['reference_depth_survivors'] = len(a)
    seed, positive = project(a @ R)
    use = positive & inside(seed)
    uv, seed = uv[use], seed[use]; counts['projected_seeds'] = len(seed)
    if not len(seed): return (*empty, counts)
    p = np.float32(uv).reshape(-1, 1, 2)
    initial = np.float32(seed).reshape(-1, 1, 2)
    options = dict(winSize=(RULES['lk_window'],) * 2, maxLevel=RULES['lk_levels'],
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .01),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    q, status, _ = cv2.calcOpticalFlowPyrLK(reference.gray, current.gray, p, initial.copy(), **options)
    if q is None or status is None: return (*empty, counts)
    good = status.ravel().astype(bool) & inside(q[:, 0])
    p, q = p[good], q[good]; counts['forward_flow_survivors'] = len(p)
    if not len(p): return (*empty, counts)
    # Do not send invalid/failed forward endpoints into reverse optical flow.
    back, status, _ = cv2.calcOpticalFlowPyrLK(current.gray, reference.gray, q, p.copy(), **options)
    if back is None or status is None: return (*empty, counts)
    good = status.ravel().astype(bool) & inside(back[:, 0])
    good &= np.linalg.norm(back[:, 0] - p[:, 0], axis=1) <= RULES['fb_pixels']
    ua, ub = p[good, 0], q[good, 0]; counts['bidirectional_flow_survivors'] = len(ua)
    keep = []; seen = set()
    for i, point in enumerate(ub):
        cell = tuple(np.rint(point * 2).astype(int))
        if cell in seen: continue
        keep.append(i); seen.add(cell)
    ua, ub = ua[keep], ub[keep]; counts['unique_current_points'] = len(ua)
    a, va = lift(reference.depth, ua); b, vb = lift(current.depth, ub)
    use = va & vb; counts['paired_depth_survivors'] = int(use.sum())
    return a[use], b[use], ua[use], ub[use], counts


class GyroSeededRGBDPose(MultiReferenceRGBDPose):
    """One fixed matcher for all references; no original/alternate mode search."""
    def __init__(self):
        super().__init__(); self.correspondence_attempts = []

    def observe(self, *args, **kwargs):
        self.correspondence_attempts = []
        return super().observe(*args, **kwargs)

    def _candidate(self, ref, current, G):
        if len(self.correspondence_attempts) >= 8:
            raise SensorContractError('bounded original reference population required')
        row = dict(reference_frame=ref.frame, status='MATCHING', counts=None, rejection=None)
        self.correspondence_attempts.append(row)
        try:
            relative = ref.gyro.T @ G
            a, b, ua, ub, counts = matched_points(ref.features, current, relative)
            row.update(status='REGISTERING', counts=counts)
            local_R, t, mask, reg = register(a, b, ua, ub, gyro_rotation=relative, mode='gyro', frame=self.frame)
            R = ref.rotation @ local_R; p = ref.position + ref.rotation @ t
            if (np.linalg.norm(p - self.last_p) > RIGID_RULES['maximum_increment_translation_m']
                    or angle(self.last_R.T @ R) > RIGID_RULES['maximum_increment_rotation_rad']):
                raise SensorContractError('consecutive rigid-pose displacement envelope rejected')
            reg |= dict(reference_frame=ref.frame, reference_measured_ns=ref.measured_ns,
                translation_reference_body_m=t.tolist(), relative_rotation=local_R.tolist(),
                gyro_relative_rotation=relative.tolist(), reference_inlier_pixels=ua[mask].tolist(),
                current_inlier_pixels=ub[mask].tolist(), reference_inlier_points_body_m=a[mask].tolist(),
                current_inlier_points_body_m=b[mask].tolist(), correspondence_counts=counts)
            row.update(status='CANDIDATE_QUALIFIED', registration={k: deepcopy(reg[k]) for k in
                ('lifted_matches', 'inliers', 'inlier_fraction', 'reference_grid_cells', 'current_grid_cells',
                 'residual_rms_m', 'reference_scatter_rms_m', 'current_scatter_rms_m', 'pruning_rounds')})
            return dict(reference=ref, R=R, p=p, local_R=local_R, t=t, registration=reg)
        except SensorContractError as error:
            row.update(status='REJECTED', rejection=str(error))
            raise


class GyroSeededVisualLedMotion(MultiReferenceVisualLedMotion):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity); self.model = GyroSeededRGBDPose()

    def observe(self, *args, **kwargs):
        self.model.correspondence_attempts = []
        result = super().observe(*args, **kwargs)
        return result | dict(correspondence_attempts=deepcopy(self.model.correspondence_attempts),
            correspondence_method='gyro_seeded_bidirectional_flow_v1',
            correspondence_checks_changed=True, rigid_registration_gates_changed=False,
            native_pose_input=False, navigation_qualified=False)
