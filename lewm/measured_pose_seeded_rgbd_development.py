"""Development-only past-measurement initialization; no pose extrapolation.

The previous accepted position seeds flow, but is not the current observation,
a new constraint, a command prior or a calibrated uncertainty certificate.
Frozen predecessor sources and all rigid/reference/motion gates are unchanged.
"""
from copy import deepcopy
import cv2
import numpy as np
from lewm.causal_sensor_state import SensorContractError, _ns
from lewm.joint_rgbd_rigid_pose_development import proper, register, angle, RIGID_RULES
from lewm.multi_reference_rgbd_pose_development import MultiReferenceRGBDPose, MultiReferenceVisualLedMotion
from lewm.rgbd_correspondence_motion_development import RULES, lift, project
from lewm.gyro_seeded_rgbd_correspondence_development import inside, validate_frame


def matched_points(reference, current, rotation, translation_seed):
    """Track measured pairs using a previous-pose translation *initial guess*.

    R maps current-body vectors to reference-body vectors. Project (a - t_seed)
    through R, including the camera lever arm. The seed never becomes an output
    pose or a rigid-fit constraint; current RGB-D must still support the fit.
    """
    R = proper(rotation)
    translation_seed = np.asarray(translation_seed, float)
    if (translation_seed.shape != (3,) or not np.isfinite(translation_seed).all()
            or np.linalg.norm(translation_seed) > RIGID_RULES['maximum_reference_translation_m']):
        raise SensorContractError('finite bounded reference-frame translation seed required')
    validate_frame(reference); validate_frame(current)
    counts = dict(reference_keypoints=len(reference.keypoints), unique_reference_points=0,
        reference_depth_survivors=0, projected_seeds=0, forward_flow_survivors=0,
        bidirectional_flow_survivors=0, unique_current_points=0, paired_depth_survivors=0,
        descriptor_association_used=False, command_translation_prior_used=False,
        gyro_seed_is_measured_correspondence=False, uncertainty_calibrated=False,
        translation_seed_reference_m=translation_seed.tolist(), seed_is_current_pose=False)
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
    seed, positive = project((a - translation_seed) @ R)
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


class MeasuredPoseSeededRGBDPose(MultiReferenceRGBDPose):
    """One fixed matcher for all references; no original/alternate mode search."""
    def __init__(self):
        super().__init__(); self.correspondence_attempts = []
        self.last_pose_ns = self.query_ns = None

    def observe(self, policy, depth, fast, *, now_ns):
        self.correspondence_attempts = []
        try:
            now = _ns(now_ns, 'measured-pose initializer query')
            if self.last_pose_ns is not None and now - self.last_pose_ns != 100_000_000:
                raise SensorContractError('immediately preceding measured pose required; no stale seed')
        except (ValueError, TypeError):
            self.failed = True
            raise
        self.query_ns = now
        result = super().observe(policy, depth, fast, now_ns=now)
        self.last_pose_ns = now
        return result

    def _candidate(self, ref, current, G):
        if len(self.correspondence_attempts) >= 8:
            raise SensorContractError('bounded original reference population required')
        row = dict(reference_frame=ref.frame, status='MATCHING', counts=None, rejection=None)
        self.correspondence_attempts.append(row)
        try:
            relative = ref.gyro.T @ G
            if (self.last_pose_ns is None or self.query_ns - self.last_pose_ns != 100_000_000
                    or ref.measured_ns > self.last_pose_ns):
                raise SensorContractError('past accepted same-anchor pose required for initializer')
            seed = ref.rotation.T @ (self.last_p - ref.position)
            row.update(seed_measured_ns=self.last_pose_ns, seed_is_current_pose=False)
            a, b, ua, ub, counts = matched_points(ref.features, current, relative, seed)
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


class MeasuredPoseSeededVisualLedMotion(MultiReferenceVisualLedMotion):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity); self.model = MeasuredPoseSeededRGBDPose()

    def observe(self, *args, **kwargs):
        self.model.correspondence_attempts = []
        result = super().observe(*args, **kwargs)
        return result | dict(correspondence_attempts=deepcopy(self.model.correspondence_attempts),
            correspondence_method='previous_measured_pose_seeded_bidirectional_flow_v1',
            correspondence_checks_changed=True, rigid_registration_gates_changed=False,
            native_pose_input=False, navigation_qualified=False)
