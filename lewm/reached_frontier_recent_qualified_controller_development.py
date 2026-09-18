"""Only observed frontier transitions over the original completed maze3 policy."""
from copy import deepcopy
import numpy as np

from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportMap
from lewm.recent_qualified_direct_flow_controller_development import RecentQualifiedDirectFlowController
from lewm.reached_frontier_state_development import ReachedFrontierState


class ReachedFrontierMap(MeasuredFloorTransportMap):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity)
        self.frontier_visits = ReachedFrontierState()

    def waypoint(self, goal_initial_xy, *, now_ns):
        original = super().waypoint(goal_initial_xy, now_ns=now_ns)
        position = self.map_from_initial@self.surface.position
        target = self.map_from_initial@np.r_[goal_initial_xy, 0.]
        return self.frontier_visits.waypoint(self.floor, self.occupied, position[:2], target[:2], original,
            mission_goal=goal_initial_xy, now_ns=now_ns)


class ReachedFrontierRecentQualifiedController(RecentQualifiedDirectFlowController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapper = ReachedFrontierMap(identity=(0, 0, 0))
        self.memory = self.mapper.surface

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | dict(controller='reached_frontier_recent_qualified_controller_v1',
            reached_frontier_transition_enabled=True,
            last_frontier_transition_receipt=deepcopy(self.mapper.frontier_visits.last_receipt))
