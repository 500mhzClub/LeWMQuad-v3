"""Use the equivalent sparse floor-candidate kernel in gyro pose estimation."""
from lewm.eligible_floor_registration_development import bind
from lewm.measured_plane_dual_camera_pose_development import MeasuredPlaneDualCameraPose
from lewm.sparse_floor_candidates_development import measured_candidates
from lewm.orthonormal_gyro_visual_motion_development import (
    OrthonormalGyroPose, OrthonormalGyroVisualMotion)


class SparseFloorGyroPose(OrthonormalGyroPose):
    _prepare_plane = bind(MeasuredPlaneDualCameraPose._prepare_plane,
        measured_candidates=measured_candidates)


class SparseFloorGyroVisualMotion(OrthonormalGyroVisualMotion):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity)
        self.model = SparseFloorGyroPose()

    def snapshot(self, *, now_ns):
        return super().snapshot(now_ns=now_ns) | dict(sparse_floor_candidate_projection=True)


def initialize_pose():
    import cv2
    import torch
    from lewm.two_cm_floor_extent_development import configure
    from lewm import process_mapped_runtime_development as process
    configure(); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    process._motion = SparseFloorGyroVisualMotion()
