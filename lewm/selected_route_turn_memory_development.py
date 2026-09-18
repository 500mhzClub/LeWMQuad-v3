"""Apply interruption memory to the turn selected after clearance handling."""
import math

import numpy as np

from lewm.clearance_turn_recovery_development import wrap
from lewm.interrupted_route_turn_memory_development import InterruptedRouteTurnMemory, TURNS


class SelectedRouteTurnMemory(InterruptedRouteTurnMemory):
    def select(self, selection, position, heading, generation, weak_trigger):
        action = selection['action']
        if (weak_trigger is None and 'scan_utilities' not in selection
                and generation == self.generation and self.active is None
                and action in TURNS):
            target = wrap(heading + math.atan2(*selection['waypoint_body_xy_m'][::-1]))
            failed = {r['direction'] for r in self.failed
                if self.nearby(r, np.asarray(position, float)[:2], target)}
            # The clearance controller may have latched the opposite of the
            # original score's preference. Consult the executed-attempt memory
            # for that selected direction, while retaining every parent gate.
            if TURNS[action] in failed and -TURNS[action] not in failed:
                revised = selection | dict(before_memory_filter_action=action)
                result = super().select(revised, position, heading, generation, weak_trigger)
                result['before_memory_filter_action'] = selection.get('before_memory_filter_action')
                if 'visual_route_turn_memory' in result:
                    result['visual_route_turn_memory'].update(
                        interruption_match_action=action,
                        interruption_match_stage='after_clearance_selection')
                return result
        return super().select(selection, position, heading, generation, weak_trigger)


class SelectedRouteTurnMemoryMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert type(self.route_turn_memory) is InterruptedRouteTurnMemory
        self.route_turn_memory = SelectedRouteTurnMemory()
