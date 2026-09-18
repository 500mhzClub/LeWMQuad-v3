"""Robust gyro-conditioned image consensus with pair-local plane rejection."""
from copy import deepcopy
import numpy as np
from lewm.gyro_conditioned_pair_pose_development import GyroConditionedPairPose, refit
from lewm.joint_rgbd_rigid_pose_development import fit, inliers
from lewm.joint_camera_registration_development import project_views
from lewm.auxiliary_reference_pose_adapter_development import body_from_reference
from lewm.conditioned_support_tracker_development import bind, CONDITIONED_RULES
from lewm.development_support_tracker_development import use
from lewm.rgbd_correspondence_motion_development import cells
from lewm.measured_plane_dual_camera_pose_development import PlaneImageConflict
from lewm.causal_sensor_state import SensorContractError
from lewm.gyro_coherent_floor_constraint_development import constrain_translation


def consensus_refit(candidate, gyro, *, camera, last_p, last_R):
    original = candidate['registration']
    names = ('reference_inlier_points_body_m', 'current_inlier_points_body_m',
        'reference_inlier_pixels', 'current_inlier_pixels')
    a, b, ua, ub = [np.asarray(original[k], float) for k in names]
    joint = original.get('calibrated_body_frame_fit', False)
    split = original['camera_inliers'][0] if joint else None
    if joint:
        check = bind(inliers, project=lambda points: project_views(points, split))
    elif camera == 'auxiliary':
        A, offset = body_from_reference()
        def check(a, b, ua, ub, R, t):
            return inliers((a-offset)@A, (b-offset)@A, ua, ub,
                A.T@R@A, A.T@(t-offset+R@offset))
    else: check = inliers
    mask = np.ones(len(a), dtype=bool); rounds = 0
    floor = original.get('measured_plane_refinement', {})
    while True:
        count = int(mask.sum())
        if count < CONDITIONED_RULES['minimum_matches'] or 2*count <= original['lifted_matches']:
            raise SensorContractError('gyro consensus requires original match count and strict majority')
        if joint and min(int(mask[:split].sum()), int(mask[split:].sum())) < 3:
            raise SensorContractError('gyro consensus needs support from both cameras')
        R, t, _ = fit(a[mask], b[mask], gyro_rotation=gyro)
        if 'gyro_coherent_floor_constraint' in original:
            t = constrain_translation(t, original['gyro_coherent_floor_constraint'], gyro)
        elif floor.get('applied', False):
            pa, pb = [floor[k]['joint_plane'] for k in ('reference_floor', 'current_floor')]
            n = np.asarray(pa['normal_body'])
            t += n*(pb['offset_body_m']-pa['offset_body_m']-n@t)
        good, _ = check(a, b, ua, ub, R, t)
        use = mask & good; rounds += 1
        if np.array_equal(use, mask): break
        mask = use
    reg = deepcopy(original)
    for name, values in zip(names, (a, b, ua, ub)):
        reg[name] = values[mask].tolist()
    reg.update(inliers=int(mask.sum()), inlier_fraction=float(mask.sum()/original['lifted_matches']),
        reference_grid_cells=cells(ua[mask]), current_grid_cells=cells(ub[mask]))
    if joint: reg['camera_inliers'] = [int(mask[:split].sum()), int(mask[split:].sum())]
    result = refit(candidate | dict(registration=reg), gyro, camera=camera,
        last_p=last_p, last_R=last_R)
    result['registration']['gyro_conditioned_refinement'].update(
        robust_gyro_image_consensus=True, all_accepted_image_points_retained=bool(mask.all()),
        all_retained_consensus_points_checked=True, initial_image_inliers=len(a),
        original_lifted_matches=original['lifted_matches'], pruning_rounds=rounds,
        retained_original_indices=np.flatnonzero(mask).tolist(),
        rejected_original_indices=np.flatnonzero(~mask).tolist(),
        strict_majority_of_original_matches=True, absolute_image_and_motion_thresholds_unchanged=True)
    return result


_candidate_with_consensus = use(GyroConditionedPairPose._candidate, refit=consensus_refit)


class GyroConsensusPairPose(GyroConditionedPairPose):
    def _refine_candidate(self, candidate, reference_plane, current_plane, **kwargs):
        try:
            return super()._refine_candidate(candidate, reference_plane, current_plane, **kwargs)
        except PlaneImageConflict as error:
            self.rejected_plane_pairs.append(dict(frame=self.frame,
                reference_frame=candidate['reference'].frame, camera=kwargs['camera'],
                reason=str(error), conflicting_pose_admitted=False))
            raise SensorContractError('image/plane pair conflicts; other measured pairs remain eligible') from error

    def _candidate(self, ref, current, G):
        result = _candidate_with_consensus(self, ref, current, G)
        reg = result['registration']
        self.rotation_measurements[-1].update({k: reg[k] for k in (
            'inliers', 'inlier_fraction', 'reference_grid_cells', 'current_grid_cells')})
        if reg.get('calibrated_body_frame_fit'):
            self.rotation_measurements[-1]['camera_inliers'] = reg['camera_inliers']
        return result

    def observe(self, *args, **kwargs):
        result = super().observe(*args, **kwargs)
        return result | dict(measured_plane_constrained_estimator=False,
            gyro_conditioned_image_consensus_estimator=True,
            raw_estimator_replay_only=True)
