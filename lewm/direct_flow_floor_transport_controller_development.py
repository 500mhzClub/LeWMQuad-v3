"""Separate observed-controller integration; no native success claim."""
from copy import deepcopy
from lewm.dual_camera_visual_motion_development import DualCameraVisualMotion
from lewm.direct_flow_dual_camera_pose_development import DirectFlowDualCameraAnchorPose
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportController


class DirectFlowDualCameraVisualMotion(DualCameraVisualMotion):
    def __init__(self, *, identity=(0,0,0)):
        super().__init__(identity=identity)
        self.model=DirectFlowDualCameraAnchorPose()

    def snapshot(self, *, now_ns):
        result=super().snapshot(now_ns=now_ns)
        if self.model.last_direct_flow_fallback is not None:
            result['direct_corner_flow_fallback']=deepcopy(self.model.last_direct_flow_fallback)
        return result


class DirectFlowFloorTransportController(MeasuredFloorTransportController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion=DirectFlowDualCameraVisualMotion(identity=(0,0,0))

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs)|dict(controller='direct_flow_floor_transport_controller_v1',
            direct_corner_flow_missingness_fallback_enabled=True)
