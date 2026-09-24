"""Inactive model-free terminal control hypothesis for the existing action bank."""
from copy import deepcopy
import math

from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.terminal_translation_pulse_development import PERIOD

HEADING_TOLERANCE_RAD = .1  # Existing measured panorama alignment tolerance.


def heading_first_terminal(selection, *, arrival_radius_m):
    if selection.get('candidate_future_outcomes_evaluated') is not False:
        raise ValueError('explicit model-free selection required')
    if selection['scan_heading_error_rad'] is not None or not selection['current_nominal_disk_clear']:
        return selection
    x, y = selection['waypoint_body_xy_m']
    if not all(math.isfinite(v) for v in (x, y, arrival_radius_m)) or arrival_radius_m <= 0:
        raise ValueError('finite observed target and positive arrival radius required')
    error = math.atan2(y, x)
    if math.hypot(x, y) <= arrival_radius_m:
        action = 'hold'
    elif abs(error) <= HEADING_TOLERANCE_RAD:
        action = 'forward'
    else:
        action = 'left_turn' if error > 0 else 'right_turn'
    row = next(r for r in selection['candidates'] if r['action'] == action)
    if not row['eligible']:
        return selection
    result = deepcopy(selection)
    result.update(action=action, action_index=ACTIONS.index(action),
        requested_command=candidate_commands(action)[0],
        command_duration_ns=PERIOD if action == 'forward' else 4*PERIOD,
        rule='terminal_measured_heading_then_existing_forward_pulse',
        heading_first_terminal=dict(previous_action=selection['action'],
            selected_action=action, changed=action != selection['action'],
            measured_heading_error_rad=error, heading_tolerance_rad=HEADING_TOLERANCE_RAD,
            arrival_radius_m=arrival_radius_m, future_state_predicted=False,
            arrival_still_requires_measured_quiet_dwell=True))
    result['terminal_translation_pulse']['selected_translation_pulse'] = action == 'forward'
    return result


class HeadingFirstTerminalReactiveMixin:
    def _select_action(self, *args, **kwargs):
        selected, correction = super()._select_action(*args, **kwargs)
        if not self.terminal_position_approach:
            return selected, correction
        if correction is not None:
            raise ValueError('terminal reactive treatment must not use a motion correction')
        return heading_first_terminal(selected,
            arrival_radius_m=self.mission.arrival_radius_m), correction
