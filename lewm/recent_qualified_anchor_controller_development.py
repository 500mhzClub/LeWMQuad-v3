"""Recent qualified visual reference inside the unchanged height controller."""
from copy import deepcopy
from lewm.recent_qualified_anchor_development import RecentQualifiedAnchorPose
from lewm.direct_flow_floor_transport_controller_development import DirectFlowDualCameraVisualMotion
from lewm.partial_floor_height_controller_development import PartialHeightDirectFlowController


class RecentQualifiedAnchorVisualMotion(DirectFlowDualCameraVisualMotion):
    def __init__(self, *, identity=(0,0,0)):
        super().__init__(identity=identity)
        self.model = RecentQualifiedAnchorPose()

    def snapshot(self, *, now_ns):
        return super().snapshot(now_ns=now_ns)|dict(
            recent_qualified_anchor=deepcopy(self.model.last_recent_qualified_anchor))


class RecentQualifiedAnchorController(PartialHeightDirectFlowController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.motion = RecentQualifiedAnchorVisualMotion(identity=(0,0,0))

    def _result(self, *args, **kwargs):
        return super()._result(*args,**kwargs)|dict(
            controller='recent_qualified_anchor_controller_v1',recent_qualified_anchor_enabled=True)
