"""Experimental gyro rotation with translation refitted to measured RGB-D pairs.

This raw estimator is for sensor replay first. It is deliberately not exposed
as the existing image-only rotation observer or selected by a native launcher.
"""
from copy import deepcopy
import numpy as np

from lewm.pair_local_plane_consensus_development import PairLocalPlaneConsensusPose
from lewm.joint_rgbd_rigid_pose_development import fit, inliers, angle, RIGID_RULES
from lewm.joint_camera_registration_development import project_views
from lewm.auxiliary_reference_pose_adapter_development import body_from_reference
from lewm.conditioned_support_tracker_development import bind
from lewm.temporal_anchor_continuity_development import CONTINUITY_RULES
from lewm.causal_sensor_state import SensorContractError
from lewm.gyro_coherent_floor_constraint_development import constrain_translation


def refit(candidate, gyro, *, camera, last_p, last_R):
    original = candidate['registration']
    a, b, ua, ub = [np.asarray(original[k], float) for k in (
        'reference_inlier_points_body_m', 'current_inlier_points_body_m',
        'reference_inlier_pixels', 'current_inlier_pixels')]
    R, t, conditioning = fit(a, b, gyro_rotation=gyro)
    floor = original.get('measured_plane_refinement', {})
    height_constrained = floor.get('applied', False)
    coherent = original.get('gyro_coherent_floor_constraint')
    if coherent is not None:
        t = constrain_translation(t, coherent, gyro)
    elif height_constrained:
        pa = floor['reference_floor']['joint_plane']
        pb = floor['current_floor']['joint_plane']
        normal = np.asarray(pa['normal_body'])
        t += normal * (pb['offset_body_m']-pa['offset_body_m']-normal@t)
    if original.get('calibrated_body_frame_fit'):
        check = bind(inliers, project=lambda points: project_views(points, original['camera_inliers'][0]))
        keep, residual = check(a, b, ua, ub, R, t)
    elif camera == 'auxiliary':
        A, offset = body_from_reference()
        keep, residual = inliers((a-offset)@A, (b-offset)@A, ua, ub,
            A.T@R@A, A.T@(t-offset+R@offset))
    else:
        keep, residual = inliers(a, b, ua, ub, R, t)
    if not keep.all():
        raise SensorContractError('gyro refit loses an accepted RGB-D correspondence')
    ref = candidate['reference']
    p, global_R = ref.position+ref.rotation@t, ref.rotation@R
    if (np.linalg.norm(t) > RIGID_RULES['maximum_reference_translation_m']
            or np.linalg.norm(p-last_p) > RIGID_RULES['maximum_increment_translation_m']
            or angle(last_R.T@global_R) > RIGID_RULES['maximum_increment_rotation_rad']
            or np.linalg.norm(p-candidate['p']) > CONTINUITY_RULES['maximum_measured_disagreement_m']
            or angle(candidate['R'].T@global_R) > CONTINUITY_RULES['maximum_measured_disagreement_rad']):
        raise SensorContractError('gyro refit conflicts with measured image pose or motion bounds')
    reg = deepcopy(original)
    # Preserve the original image/plane fit as evidence, never as a description
    # of the new gyro-conditioned fit.
    reg.update(mode='gyro_rgbd_refit', relative_rotation=R.tolist(),
        translation_reference_body_m=t.tolist(), gyro_disagreement_rad=angle(gyro.T@R),
        residual_rms_m=float(np.sqrt(np.mean(residual**2))),
        gyro_conditioned_refinement=dict(conditioning=conditioning,
            all_accepted_image_points_retained=True, translation_from_image_correspondences=True,
            measured_floor_offset_constraint_used=height_constrained,
            gyro_coherent_floor_constraint_used=coherent is not None,
            original_relative_rotation=candidate['local_R'].tolist(),
            original_translation_m=candidate['t'].tolist(),
            original_image_plane_refinement=reg.pop('measured_plane_refinement', None),
            original_image_gyro_disagreement_rad=angle(gyro.T@candidate['local_R']),
            gyro_bias_estimated=False, command_translation_integration_used=False))
    return dict(reference=ref, R=global_R, p=p, local_R=R, t=t, registration=reg)


class GyroConditionedPairPose(PairLocalPlaneConsensusPose):
    def _candidate(self, ref, current, G):
        start = len(self.rotation_measurements)
        candidate = super()._candidate(ref, current, G)
        try:
            result = refit(candidate, ref.gyro.T@G, camera=self.camera,
                last_p=self.last_p, last_R=self.last_R)
            reg = result['registration']
            self.rotation_measurements[-1].update(
                fitted_rotation_reference_body_from_current_body=result['local_R'].tolist(),
                composed_rotation_initial_body_from_current_body=result['R'].tolist(),
                position_initial_body_m=result['p'].tolist(), fitting_mode=reg['mode'],
                residual_rms_m=reg['residual_rms_m'], gyro_disagreement_rad=reg['gyro_disagreement_rad'],
                measured_plane_constrained=False, gyro_conditioned_rgbd_translation=True)
            return result
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError):
            del self.rotation_measurements[start:]
            raise

    def observe(self, *args, **kwargs):
        result = super().observe(*args, **kwargs)
        self.last_continuity.update(rotation_fitting_mode='gyro_rgbd_refit',
            gyro_role='rotation_estimator')
        return result | dict(mode='gyro_rgbd_refit', gyro_role='rotation_estimator',
            gyro_bias_estimated=False, raw_estimator_replay_only=True)
