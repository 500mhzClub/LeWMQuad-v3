"""Prospective recovery variant: rejoin a clear improving preferred turn."""
from copy import deepcopy
import math

from lewm.arrival_entry_terminal_priority_development import ArrivalEntryTerminalPriorityRuntime
from lewm.clearance_turn_recovery_development import choose
from lewm.geometry_progress_pilot_development import ACTIONS

TURNS = ('left_turn', 'right_turn')


def release_heading_turn(selection, state):
    latch = selection.get('clearance_turn') or {}
    preferred = selection.get('before_memory_filter_action')
    current = selection['action']
    if (state is None or not latch.get('active') or preferred not in TURNS
            or current not in TURNS or preferred == current):
        return selection, state
    latched_action = 'left_turn' if state['direction'] == 1 else 'right_turn'
    if current != latched_action:
        return selection, state
    clearance = next(r for r in selection['memory_forecast_candidates'] if r['action'] == preferred)
    if not clearance['full_reserve_path_clear']:
        return selection, state
    utilities = {r['action']:r['utility_m'] for r in selection.get('scan_utilities', selection['candidates'])}
    if (not all(math.isfinite(utilities[a]) for a in (preferred, current, 'hold'))
            or utilities[preferred] <= utilities[current]):
        return selection, state
    if 'scan_utilities' in selection:
        improving = utilities[preferred] > utilities['hold']
    else:
        candidate = next(r for r in selection['candidates'] if r['action'] == preferred)
        improving = (candidate['predicted_heading_error_at_commit_end_rad']
            < candidate['predicted_heading_error_at_commit_start_rad'])
    if not improving:
        return selection, state
    result = deepcopy(selection)
    choose(result, ACTIONS.index(preferred))
    result['clearance_turn'] = dict(active=False,
        event='FULL_RESERVE_PREFERRED_HEADING_REJOINS_ROUTE_OR_VIEW',
        previous_target_heading_rad=state['target_heading_rad'],
        previous_direction=state['direction'], released_action=preferred)
    result['full_reserve_heading_release'] = dict(applied=True,
        previous_action=current, selected_action=preferred,
        full_reserve_clearance_required=True, predicted_heading_improvement_required=True,
        previous_utility_m=utilities[current], selected_utility_m=utilities[preferred])
    return result, None


class FullReserveHeadingReleaseRuntime(ArrivalEntryTerminalPriorityRuntime):
    def _select_clear_prediction(self, *args, **kwargs):
        result = super()._select_clear_prediction(*args, **kwargs)
        if self.terminal_position_approach:
            return result
        result, self.clearance_turn = release_heading_turn(result, self.clearance_turn)
        return result
