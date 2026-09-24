"""Joint-view retained-anchor fallback inside existing plane and continuity checks."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
from lewm.conditioned_support_tracker_development import _Chained, bind
from lewm.conditioned_support_150_tracker_development import ConditionedSupport150Pose, ConditionedSupport150VisualMotion
from lewm.batched_patch_tracker_development import chained_points
from lewm.keyframe_rgbd_pose_development import matched_points
from lewm.chained_corner_flow_association_development import CHAIN_RULES
from lewm.joint_camera_registration_development import register_views, project_views
from lewm.joint_rgbd_rigid_pose_development import SensorContractError, RIGID_RULES, angle, inliers
from lewm.optional_plane_refinement_development import refine_if_supported
from lewm import measured_plane_dual_camera_pose_development as plane


def refine_joint_if_supported(candidate, reference_plane, current_plane, **kwargs):
    reg = candidate['registration']
    if not reg.get('calibrated_body_frame_fit'):
        return refine_if_supported(candidate, reference_plane, current_plane, **kwargs)
    split = reg['camera_inliers'][0]
    checked_inliers = bind(inliers, project=lambda points: project_views(points, split))
    refined = bind(plane.refine, inliers=checked_inliers)
    namespace = SimpleNamespace(**(vars(plane) | dict(refine=refined)))
    optional = bind(refine_if_supported, original=namespace)
    # Pooled inputs and fitted transform are already expressed in body coordinates.
    return optional(candidate, reference_plane, current_plane, **(kwargs | dict(camera='primary')))


class _JointCameras(_Chained):
    def _candidate(self, ref, current, G):
        previous_pair = ref is self.previous and self.frame-ref.frame == 1
        try:
            return super()._candidate(ref, current, G)
        except SensorContractError:
            # Preserve the original front-first/auxiliary result before
            # extending a missing consecutive-frame measurement to both views.
            if not (previous_pair and self.camera == 'auxiliary') and (not self._chain_mode or not any(ref is r for r in self.references)
                    or not 2 <= self.frame-ref.frame <= CHAIN_RULES['maximum_intervals']):
                raise
        fitted = None; associations = []
        if previous_pair:
            views = [matched_points(ref.features[camera], current[camera]) for camera in ('primary', 'auxiliary')]
            try:
                fitted = register_views(*views, gyro_rotation=ref.gyro.T@G, frame=self.frame)
            except SensorContractError:
                pass
        if fitted is None:
            views = []
            for camera in ('primary', 'auxiliary'):
                sequence = [(i, self._image_history[i][0], self._image_history[i][1][camera])
                    for i in range(ref.frame, self.frame+1)]
                sequence[0] = (ref.frame, ref.measured_ns, ref.features[camera])
                values, association = chained_points(sequence)
                views.append(values); associations.append(association)
            fitted = register_views(*views, gyro_rotation=ref.gyro.T@G, frame=self.frame)
        local_R, t, mask, reg, arrays = fitted
        R, p = ref.rotation@local_R, ref.position+ref.rotation@t
        if (np.linalg.norm(p-self.last_p) > RIGID_RULES['maximum_increment_translation_m']
                or angle(self.last_R.T@R) > RIGID_RULES['maximum_increment_rotation_rad']):
            raise SensorContractError('joint-camera consecutive displacement envelope rejected')
        a, b, ua, ub = arrays
        reg.update(reference_frame=ref.frame, reference_measured_ns=ref.measured_ns,
            translation_reference_body_m=t.tolist(), relative_rotation=local_R.tolist(),
            gyro_relative_rotation=(ref.gyro.T@G).tolist(),
            reference_inlier_pixels=ua[mask].tolist(), current_inlier_pixels=ub[mask].tolist(),
            reference_inlier_points_body_m=a[mask].tolist(), current_inlier_points_body_m=b[mask].tolist(),
            camera='joint', joint_camera_measurement=True,
            joint_camera_retained_anchor=any(ref is r for r in self.references),
            joint_camera_previous_frame=previous_pair,
            joint_camera_association='chained_flow' if associations else 'descriptor')
        if associations:
            reg['chained_corner_flow_association'] = dict(views=associations)
        if len(self.rotation_measurements) >= 9:
            raise SensorContractError('bounded eight anchors plus one increment required')
        self.rotation_measurements.append(dict(reference_frame=ref.frame,
            reference_measured_ns=ref.measured_ns, current_frame=self.frame,
            reference_rotation_initial_body_from_reference_body=ref.rotation.tolist(),
            fitted_rotation_reference_body_from_current_body=local_R.tolist(),
            composed_rotation_initial_body_from_current_body=R.tolist(),
            gyro_rotation_reference_body_from_current_body=(ref.gyro.T@G).tolist(),
            position_initial_body_m=p.tolist(), fitting_mode=reg['mode'],
            inliers=reg['inliers'], inlier_fraction=reg['inlier_fraction'],
            reference_grid_cells=reg['reference_grid_cells'], current_grid_cells=reg['current_grid_cells'],
            residual_rms_m=reg['residual_rms_m'], gyro_disagreement_rad=reg['gyro_disagreement_rad'],
            candidate_envelope_passed=True, witness_alone_grants_pose=False, camera='joint',
            camera_inliers=reg['camera_inliers'], camera_specific_reprojection=True))
        if self.last_chained_anchor_fallback is not None:
            self.last_chained_anchor_fallback['pair_attempts'].append(dict(camera='joint',
                reference_frame=ref.frame, current_frame=self.frame, qualified=True,
                camera_inliers=reg['camera_inliers']))
        return dict(reference=ref, R=R, p=p, local_R=local_R, t=t, registration=reg)


class JointCameraAnchorPose(ConditionedSupport150Pose, _JointCameras):
    _refine_candidate = staticmethod(refine_joint_if_supported)

    def observe(self, *args, **kwargs):
        result = super().observe(*args, **kwargs)
        if (result['registration'] or {}).get('joint_camera_measurement'):
            self.last_camera_selection.update(search_camera=self.camera, selected_camera='joint',
                auxiliary_attempted=True, joint_camera_measurement=True,
                joint_camera_retained_anchor=result['registration']['joint_camera_retained_anchor'])
            result['camera_selection'] = deepcopy(self.last_camera_selection)
        return result


class JointCameraAnchorVisualMotion(ConditionedSupport150VisualMotion):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity)
        self.model = JointCameraAnchorPose()

    def snapshot(self, *, now_ns):
        from lewm import joint_measured_floor_plane_development as floor
        return super().snapshot(now_ns=now_ns) | dict(joint_camera_retained_anchor_fallback=True,
            joint_camera_previous_frame_fallback=True,
            floor_minimum_second_extent_m=floor.MINIMUM_SECOND_EXTENT_M,
            original_global_floor_gate_unchanged=floor.MINIMUM_SECOND_EXTENT_M == .05)


def initialize_joint_camera_pose():
    import cv2
    import torch
    from lewm import process_mapped_runtime_development as process
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    process._motion = JointCameraAnchorVisualMotion()
