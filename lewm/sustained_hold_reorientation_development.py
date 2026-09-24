"""Measured, freshly checked continuation of the existing hold-reorientation turn."""
from copy import deepcopy
import math
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.hold_reorientation_development import HoldReorientation

MAX_TURN_COMMANDS = 8  # Original prediction bank contains eight 100 ms intervals.


def heading(value):
    value = np.asarray(value, float)
    if value.shape != (2,) or not np.isfinite(value).all() or np.linalg.norm(value) < .2:
        raise ValueError('current admitted nonvertical map heading required')
    return value/np.linalg.norm(value)


class SustainedHoldReorientation(HoldReorientation):
    def __init__(self):
        super().__init__()
        self.active = None

    def reset_goal(self):
        super().reset_goal()
        self.active = None

    def reconsider(self, selection, *, frame, now_ns, observed_heading_map):
        current = heading(observed_heading_map)
        if self.last_frame is not None and frame != self.last_frame+1:
            self.active = None
        active = self.active
        # The frozen helper validates the complete current six-action bank,
        # scores, original hold optimum, clock, residual and all original vetoes.
        original = super().reconsider(selection, frame=frame, now_ns=now_ns)
        if active is not None:
            start = np.asarray(active['start_heading_map'])
            signed = math.atan2(start[0]*current[1]-start[1]*current[0], float(start@current))
            progress = active['direction']*signed
            if (self.holds == 0 or original is not selection
                    or active['commands'] >= MAX_TURN_COMMANDS
                    or progress >= active['target_observed_turn_rad']):
                self.active = None; self.holds = 0
                return original
            index = ACTIONS.index(active['action'])
            if (active['action'] not in selection['phase_allowed_actions']
                    or selection['surface_checks'][index]['possible_intersection']
                    or not selection['nominal_path_checks'][index]['all_predicted_segments_nominally_clear']):
                self.active = None; self.holds = 0
                return original
            result = deepcopy(selection)
            result.update(action=active['action'], action_index=index,
                requested_command=list(candidate_commands(active['action'])[0]))
            active['commands'] += 1
            self.holds = 0
            return self._receipt(result, frame, now_ns, progress, starting=False)
        event = original.get('hold_reorientation')
        if event is None:
            return original
        index = ACTIONS.index(event['selected_action'])
        p = np.asarray(original['prediction'], float)
        if np.hypot(p[index,-1,2],p[index,-1,3]) <= 1e-8:
            raise ValueError('defined original full-horizon yaw target required')
        direction = 1 if event['selected_action']=='left_turn' else -1
        target = direction*math.atan2(p[index,-1,2],p[index,-1,3])
        if not 0 < target < math.pi:
            # Preserve the original single turn; do not invent a target from a
            # model forecast that does not predict the selected turn direction.
            return original
        self.active = dict(start_frame=frame, start_heading_map=current.tolist(),
            action=event['selected_action'], direction=direction,
            target_observed_turn_rad=target, commands=1)
        return self._receipt(original, frame, now_ns, 0., starting=True)

    def _receipt(self, selection, frame, now_ns, progress, *, starting):
        return selection | dict(sustained_hold_reorientation=dict(
            frame=frame, measured_ns=now_ns, starting=starting, **deepcopy(self.active),
            observed_turn_progress_rad=progress, maximum_turn_commands=MAX_TURN_COMMANDS,
            target_source='original_selected_action_800ms_predicted_yaw',
            current_six_action_forecast_and_all_original_vetoes_checked=True,
            original_forecasts_scores_and_vetoes_preserved=True,
            geometry_veto_relaxed=False, command_horizon_ns=100_000_000,
            next_observation_required=True, native_state_used=False,
            physical_clearance_certified=False, goal_achieved=False))
