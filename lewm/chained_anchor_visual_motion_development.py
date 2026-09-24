"""Expose the separate chained-anchor observer through the existing public contract."""
from copy import deepcopy

from lewm.direct_flow_floor_transport_controller_development import DirectFlowDualCameraVisualMotion
from lewm.chained_anchor_dual_camera_pose_development import ChainedAnchorDualCameraPose


class ChainedAnchorVisualMotion(DirectFlowDualCameraVisualMotion):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity)
        self.model = ChainedAnchorDualCameraPose()

    def snapshot(self, *, now_ns):
        result = super().snapshot(now_ns=now_ns)
        if self.model.last_chained_anchor_fallback is not None:
            result['chained_anchor_fallback'] = deepcopy(self.model.last_chained_anchor_fallback)
        return result
