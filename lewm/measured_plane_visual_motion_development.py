"""Explicit estimator label over the unchanged dual-camera public witness contract."""
from copy import deepcopy
from lewm.dual_camera_visual_motion_development import DualCameraVisualMotion
from lewm.measured_plane_dual_camera_pose_development import MeasuredPlaneDualCameraPose


class MeasuredPlaneVisualMotion(DualCameraVisualMotion):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity)
        self.model = MeasuredPlaneDualCameraPose()

    def snapshot(self, *, now_ns):
        original = super().snapshot(now_ns=now_ns)
        return original | dict(
            measured_plane_constrained_estimator=True,
            measured_plane_evidence=deepcopy(self.model.last_measured_plane),
            measured_plane_evidence_current=original['current_pose'] is not None,
            measured_plane_selected_pair=deepcopy(self.model.last_measured_plane_refinement),
            original_global_floor_gate_unchanged=True,
            original_temporal_gate_values_unchanged=True, reference_history_reset=False)
