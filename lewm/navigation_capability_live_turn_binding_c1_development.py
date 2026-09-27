"""Release a visually motivated turn latch when its direction becomes blocked."""
from lewm.interrupted_route_turn_memory_development import InterruptedRouteTurnMemory
from lewm.navigation_capability_exhausted_view_development import ExhaustedViewRuntimeMixin


class LiveEligibleRouteTurnMemory(InterruptedRouteTurnMemory):
    def select(self, selection, position, heading, generation, weak_trigger):
        released = None
        if (self.active is not None and self.generation == generation
                and weak_trigger is None and 'scan_utilities' not in selection):
            action = 'left_turn' if self.active['direction'] == 1 else 'right_turn'
            row = next(r for r in selection['memory_forecast_candidates'] if r['action'] == action)
            if not row['nominal_predicted_path_clear']:
                released = dict(action=action,target_heading_rad=self.active['target_heading_rad'],
                    reason='LATCHED_DIRECTION_NO_LONGER_ELIGIBLE',clearance_rules_unchanged=True)
                self.active = self.attempt = None
        result = super().select(selection, position, heading, generation, weak_trigger)
        if released is not None:
            result = result | dict(released_visual_route_turn_memory=released)
        return result


class LiveTurnMemoryRuntimeMixin(ExhaustedViewRuntimeMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert type(self.route_turn_memory) is InterruptedRouteTurnMemory
        assert self.route_turn_memory.active is None
        self.route_turn_memory = LiveEligibleRouteTurnMemory()
