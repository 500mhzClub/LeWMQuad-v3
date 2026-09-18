"""Reuse exact segment clearances for identical observed obstacle geometry."""
from functools import lru_cache
from lewm.eligible_floor_registration_development import bind
from lewm.fine_stored_obstacle_routing_development import cached_clearance
from lewm.fine_goal_route_development import fine_goal_route, FineGoalRouteRuntime
from lewm.hold_relative_clearance_recovery_development import HoldRelativeClearanceRecoveryRuntime


class CachedSegmentClearance:
    def __init__(self, cells):
        self.geometry = cached_clearance(cells)
        self._minimum = lru_cache(maxsize=16384)(self.geometry.minimum)

    def minimum(self, start, end):
        # Exact coordinates, without rounding or reuse across obstacle changes.
        return self._minimum(tuple(start), tuple(end))


@lru_cache(maxsize=2)
def cached_graph_geometry(cells):
    return CachedSegmentClearance(cells)


cached_fine_goal_route = bind(fine_goal_route, cached_clearance=cached_graph_geometry)


class CachedFineGoalRecoveryRuntime(HoldRelativeClearanceRecoveryRuntime):
    def _routing_proposer(self, snapshot):
        original = super(FineGoalRouteRuntime, self)._routing_proposer(snapshot)
        def propose(floor, occupied, position, goal, **kwargs):
            route = original(floor, occupied, position, goal, **kwargs)
            return cached_fine_goal_route(snapshot, position, goal, route)
        return propose
