"""Separate tracker/registration variant using locally estimated floor depth."""
from lewm.eligible_floor_registration_development import bind
from lewm.local_inverse_depth_floor_development import measured_candidates, WINDOW
from lewm.measured_plane_dual_camera_pose_development import MeasuredPlaneDualCameraPose
from lewm.partial_floor_height_development import PartialHeightRegistration
from lewm.stable_gyro_reference_development import (
    CompiledFloorStableGyroReferencePose, CompiledFloorStableGyroReferenceMotion)

_prepare = bind(MeasuredPlaneDualCameraPose._prepare_plane,
    measured_candidates=measured_candidates)
_register = bind(PartialHeightRegistration.observe, measured_candidates=measured_candidates)
DESCRIPTION = dict(floor_candidate_depth_source='local_inverse_depth_average',
    floor_depth_window_pixels=WINDOW, floor_candidates_are_raw_pixel_depth=False,
    invalid_depth_filled=False, image_feature_depth_changed=False,
    absolute_floor_coherence_threshold_changed=False)


class LocalInverseDepthFloorPose(CompiledFloorStableGyroReferencePose):
    def _prepare_plane(self, current, G, now):
        receipt = _prepare(self, current, G, now)
        receipt.update(DESCRIPTION)
        return receipt


class LocalInverseDepthFloorMotion(CompiledFloorStableGyroReferenceMotion):
    def __init__(self, *, identity=(0, 0, 0), activation_frame=0):
        super().__init__(identity=identity, activation_frame=activation_frame)
        self.model = LocalInverseDepthFloorPose(activation_frame=activation_frame)

    def snapshot(self, *, now_ns):
        return super().snapshot(now_ns=now_ns) | DESCRIPTION


class LocalInverseDepthFloorRegistration(PartialHeightRegistration):
    def observe(self, *args, **kwargs):
        return _register(self, *args, **kwargs) | DESCRIPTION
