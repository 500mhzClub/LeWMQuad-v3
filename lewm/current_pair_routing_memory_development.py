"""Inactive routing-memory treatment; retain accumulated action-clearance evidence."""
from dataclasses import dataclass, fields, replace

from lewm.fine_stored_obstacle_routing_development import FineStoredMap
from lewm.multirate_routing_map_development import RoutingSnapshot


class CapturedUpdateSet(set):
    """Capture observed values, including cells already in the accumulated set."""
    def __init__(self, values=()):
        super().__init__(values)
        self.observed = set()

    def begin_observation(self):
        self.observed.clear()

    def update(self, values):
        values = set(values)
        self.observed.update(values)
        super().update(values)


@dataclass(frozen=True)
class CurrentPairRoutingSnapshot(RoutingSnapshot):
    current_floor: frozenset = frozenset()
    current_occupied: frozenset = frozenset()
    current_fine_occupied: frozenset = frozenset()

    def current_pair_view(self):
        if (not self.current_floor <= self.floor
                or not self.current_occupied <= self.occupied
                or not self.current_fine_occupied <= self.fine_occupied):
            raise ValueError('current observation must be within accumulated evidence')
        return replace(self, floor=self.current_floor, occupied=self.current_occupied,
            fine_occupied=self.current_fine_occupied)


class CapturedCurrentPairMap(FineStoredMap):
    """Run unchanged paired geometry once, capturing its existing set updates."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for name in ('floor', 'occupied', 'fine_occupied'):
            setattr(self, name, CapturedUpdateSet(getattr(self, name)))

    def update(self, *args, **kwargs):
        for cells in (self.floor, self.occupied, self.fine_occupied):
            cells.begin_observation()
        snapshot = super().update(*args, **kwargs)
        self.latest = CurrentPairRoutingSnapshot(
            **{f.name:getattr(snapshot, f.name) for f in fields(RoutingSnapshot)},
            current_floor=frozenset(self.floor.observed),
            current_occupied=frozenset(self.occupied.observed),
            current_fine_occupied=frozenset(self.fine_occupied.observed))
        return self.latest


def initialize_mapping():
    from lewm import two_cm_floor_extent_development as floor
    from lewm import process_mapped_runtime_development as process
    floor.configure()
    process.initialize_mapping()
    process._mapper = CapturedCurrentPairMap()


class RoutingMemoryScopeMixin:
    """Both prospective arms use captured snapshots and retain other state."""
    routing_memory_scope = 'persistent'

    def _routing_view(self, snapshot):
        if not isinstance(snapshot, CurrentPairRoutingSnapshot):
            raise ValueError('explicit captured paired-observation snapshot required')
        if self.routing_memory_scope == 'persistent':
            return snapshot
        if self.routing_memory_scope == 'latest_mapped_pair':
            return snapshot.current_pair_view()
        raise ValueError('fixed routing-memory condition required')

    def _route(self, snapshot, *args, **kwargs):
        return super()._route(self._routing_view(snapshot), *args, **kwargs)

    def _route_target(self, route, snapshot, position):
        return super()._route_target(route, self._routing_view(snapshot), position)

    def _select_action(self, packet, evidence, prefix, goal_body, scan_error, snapshot, q, Q):
        # The original accumulated map reaches predictive action clearance.
        selected, correction = super()._select_action(
            packet, evidence, prefix, goal_body, scan_error, snapshot, q, Q)
        view = self._routing_view(snapshot)
        selected = selected | dict(routing_memory_scope=dict(
            condition=self.routing_memory_scope, mapped_frame=snapshot.frame,
            mapped_measured_ns=snapshot.measured_ns,
            routing_floor_cells=len(view.floor), retained_floor_cells=len(snapshot.floor),
            routing_fine_obstacle_cells=len(view.fine_occupied),
            retained_fine_obstacle_cells=len(snapshot.fine_occupied),
            accumulated_action_clearance_preserved=True,
            tracker_temporal_model_mission_and_frontier_state_retained=True,
            latest_mapped_pair_is_not_necessarily_current_planning_frame=True,
            memoryless_controller=False))
        return selected, correction
