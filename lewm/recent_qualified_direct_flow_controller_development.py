"""Only recent qualified visual-reference retention over the direct-flow base."""
from lewm.direct_flow_floor_transport_controller_development import DirectFlowFloorTransportController
from lewm.recent_qualified_anchor_controller_development import RecentQualifiedAnchorVisualMotion


class RecentQualifiedDirectFlowController(DirectFlowFloorTransportController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion = RecentQualifiedAnchorVisualMotion(identity=(0, 0, 0))

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | dict(
            controller='recent_qualified_direct_flow_controller_v1',
            recent_qualified_anchor_enabled=True)
