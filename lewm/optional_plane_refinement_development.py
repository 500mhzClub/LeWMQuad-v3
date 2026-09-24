"""Retain a qualified RGB-D fit when optional plane refinement loses image support."""
from copy import deepcopy
import numpy as np
from lewm import measured_plane_dual_camera_pose_development as original
from lewm.feature_budget_100_tracker_development import FeatureBudget100Pose,FeatureBudget100VisualMotion


def refine_if_supported(candidate, reference_plane, current_plane, **kwargs):
    try:
        return original.refine(candidate, reference_plane, current_plane, **kwargs)
    except original.SensorContractError as error:
        if str(error) != 'plane refinement loses an original image inlier':
            raise
    # The original refinement checks image support before checking conflict.
    # Perform those conflict checks here before retaining the original fit.
    reg = candidate['registration']
    a = np.asarray(reg['reference_inlier_points_body_m'])
    b = np.asarray(reg['current_inlier_points_body_m'])
    R, t, _ = original.fit(a, b, reference_plane, current_plane)
    ref = candidate['reference']
    global_R, global_p = ref.rotation@R, ref.position+ref.rotation@t
    displacement = float(np.linalg.norm(global_p-candidate['p']))
    rotation = original.angle(candidate['R'].T@global_R)
    if (displacement > original.CONTINUITY_RULES['maximum_measured_disagreement_m']
            or rotation > original.CONTINUITY_RULES['maximum_measured_disagreement_rad']):
        raise original.PlaneImageConflict('qualified image and measured-plane estimates conflict')
    if (np.linalg.norm(t) > original.RIGID_RULES['maximum_reference_translation_m']
            or np.linalg.norm(global_p-kwargs['last_p']) > original.RIGID_RULES['maximum_increment_translation_m']
            or original.angle(kwargs['last_R'].T@global_R) > original.RIGID_RULES['maximum_increment_rotation_rad']
            or original.angle(kwargs['gyro'].T@R) > original.RIGID_RULES['maximum_gyro_disagreement_rad']):
        raise original.SensorContractError('plane-refined pose fails original motion or gyro gates')
    # The retained image estimate must also agree with current measured height.
    na = np.asarray(reference_plane['normal_body'])
    offset_residual = abs(float(na@candidate['t']-current_plane['offset_body_m']+reference_plane['offset_body_m']))
    if offset_residual > .003:
        raise original.PlaneImageConflict('retained image fit conflicts with measured floor height')
    retained = deepcopy(reg)
    retained['measured_plane_refinement'] = dict(applied=False,
        reason='refinement_loses_image_support_original_qualified_fit_retained',
        original_image_fit_retained=True, original_inliers_preserved=True,
        original_image_gate_values_unchanged=True, floor_image_conflict_checked=True,
        original_image_pose_disagreement_m=displacement,
        original_image_pose_disagreement_rad=rotation,
        retained_image_floor_offset_residual_m=offset_residual)
    return candidate|dict(registration=retained)


class OptionalPlanePose(FeatureBudget100Pose):
    _refine_candidate = staticmethod(refine_if_supported)


class OptionalPlaneVisualMotion(FeatureBudget100VisualMotion):
    def __init__(self, *, identity=(0,0,0)):
        super().__init__(identity=identity)
        self.model=OptionalPlanePose()


def initialize_optional_pose():
    import cv2
    import torch
    from lewm import process_mapped_runtime_development as process
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    process._motion=OptionalPlaneVisualMotion()


def initialize_optional_pose_150():
    from lewm.feature_budget_150_tracker_development import initialize_pose_150
    from lewm import process_mapped_runtime_development as process
    initialize_pose_150()
    process._motion.model._refine_candidate=refine_if_supported
