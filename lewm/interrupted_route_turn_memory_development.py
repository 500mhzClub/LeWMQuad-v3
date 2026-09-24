"""Try the other clear route-turn direction after measured visual interruption."""
from copy import deepcopy
import math

import numpy as np

from lewm.clearance_turn_recovery_development import choose, wrap
from lewm.geometry_progress_pilot_development import ACTIONS

TURNS = {'left_turn': 1, 'right_turn': -1}


class InterruptedRouteTurnMemory:
    def __init__(self):
        self.generation = None
        self.attempt = None
        self.failed = []
        self.active = None
        self.last_trigger = -1

    @staticmethod
    def nearby(record, position, target):
        return (np.linalg.norm(position - record['position']) <= .20
            and abs(wrap(target - record['target_heading_rad'])) <= .30)

    def select(self, selection, position, heading, generation, weak_trigger):
        position = np.asarray(position, float)[:2]
        if generation != self.generation:
            self.attempt = self.active = None
            self.failed = []
            self.generation = generation
        if weak_trigger is not None:
            if weak_trigger > self.last_trigger:
                self.last_trigger = weak_trigger
                attempt = self.attempt
                if (attempt is not None and np.linalg.norm(position-attempt['position']) <= .20
                        and attempt['direction'] * wrap(heading-attempt['start_heading_rad']) >= .025):
                    self.failed.append(attempt | dict(trigger_ns=weak_trigger))
                    self.failed = self.failed[-16:]
                self.attempt = self.active = None
            return selection
        if 'scan_utilities' in selection:
            self.attempt = self.active = None
            return selection
        error = math.atan2(*selection['waypoint_body_xy_m'][::-1])
        target = wrap(heading + error)
        if self.active is not None:
            remaining = (self.active['direction'] * wrap(self.active['target_heading_rad']-heading)) % (2*math.pi)
            crossed = remaining-self.active['previous_remaining_rad'] > math.pi
            if (not self.nearby(self.active, position, target) or abs(error) <= .10 or crossed):
                self.active = None
            else:
                self.active['previous_remaining_rad'] = remaining
        by_action = {r['action']: r for r in selection['memory_forecast_candidates']}
        # Clear translating progress can leave the local turn problem normally.
        if selection['action'] in ('forward', 'left_arc', 'right_arc'):
            self.attempt = self.active = None
            return selection
        failed = {r['direction'] for r in self.failed if self.nearby(r, position, target)}
        preferred = selection.get('before_memory_filter_action')
        if self.active is None and preferred in TURNS and abs(error) > .10:
            direction = -TURNS[preferred]
            alternative = 'left_turn' if direction == 1 else 'right_turn'
            if (TURNS[preferred] in failed and direction not in failed
                    and by_action[alternative]['nominal_predicted_path_clear']):
                self.active = dict(position=position.copy(), target_heading_rad=target,
                    direction=direction, previous_remaining_rad=(direction*error) % (2*math.pi))
        result = selection
        if self.active is not None:
            action = 'left_turn' if self.active['direction'] == 1 else 'right_turn'
            eligible = by_action[action]['nominal_predicted_path_clear']
            result = deepcopy(selection)
            choose(result, ACTIONS.index(action if eligible else 'hold'))
            result['visual_route_turn_memory'] = dict(active=True,
                direction=self.active['direction'], target_heading_rad=self.active['target_heading_rad'],
                failed_directions=sorted(failed), selected_turn_forecast_clear=eligible,
                visual_recovery_and_clearance_thresholds_unchanged=True)
        action = result['action']
        if action in TURNS:
            if (self.attempt is None or self.attempt['direction'] != TURNS[action]
                    or not self.nearby(self.attempt, position, target)):
                self.attempt = dict(position=position.copy(), target_heading_rad=target,
                    direction=TURNS[action], start_heading_rad=heading)
        elif self.active is None:
            self.attempt = None
        return result


class InterruptedRouteTurnMemoryMixin:
    def __init__(self, *args, **kwargs):
        self.route_turn_memory = InterruptedRouteTurnMemory()
        self.route_turn_weak_trigger = None
        super().__init__(*args, **kwargs)

    def _route(self, snapshot, evidence, goal, *, measured_ns):
        result = super()._route(snapshot, evidence, goal, measured_ns=measured_ns)
        active = evidence.get('visual_support', {}).get('recovery_state_at_observation')
        self.route_turn_weak_trigger = (active['trigger_ns'] if active is not None
            and result['status'] == 'LOW_VISUAL_SUPPORT_REQUIRES_MEASURED_VIEW' else None)
        return result

    def _select_clear_prediction(self, selected, prediction, snapshot, position, rotation):
        result = super()._select_clear_prediction(selected, prediction, snapshot, position, rotation)
        if self.terminal_position_approach:
            return result
        return self.route_turn_memory.select(result, position, math.atan2(rotation[1, 0], rotation[0, 0]),
            self.mission_generation, self.route_turn_weak_trigger)
