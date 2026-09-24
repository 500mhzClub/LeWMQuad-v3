"""Prospective survey recovery through a full-reserve translating forecast."""
from copy import deepcopy
import math

from lewm.clearance_turn_recovery_development import choose
from lewm.full_reserve_heading_release_development import FullReserveHeadingReleaseRuntime
from lewm.geometry_progress_pilot_development import ACTIONS


def reposition_survey(selection):
    latch = selection.get('clearance_turn') or {}
    preferred = selection.get('before_memory_filter_action')
    turns = ('left_turn', 'right_turn')
    if ('scan_utilities' not in selection or not latch.get('active')
            or preferred not in turns or selection['action'] not in turns
            or selection['action'] == preferred):
        return selection
    rows = {r['action']:r for r in selection['memory_forecast_candidates']}

    def finite(values):
        return len(values) == 8 and all(v is not None and math.isfinite(v) for v in values)

    hold = rows['hold']['segment_clearances_m']
    if not finite(hold) or min(hold) <= .45:
        return selection
    eligible = []
    for action in ('forward', 'left_arc', 'right_arc'):
        row = rows[action]; path = row['segment_clearances_m']
        gain = max(.001, .1*max(0., row['required_path_clearance_m']-min(hold)))
        if (row['full_reserve_path_clear'] and finite(path)
                and min(path) > row['required_path_clearance_m']+1e-12
                and min(path) >= min(hold) and path[-1] >= hold[-1]+gain):
            eligible.append((min(path), path[-1], -ACTIONS.index(action), gain))
    if not eligible:
        return selection
    minimum, endpoint, negative_index, gain = max(eligible)
    result = deepcopy(selection)
    choose(result, -negative_index)
    result['clearance_turn'] = dict(active=False,
        event='FULL_RESERVE_SURVEY_REPOSITION', previous_latch=latch)
    result['survey_clearance_reposition'] = dict(applied=True,
        previous_action=selection['action'], selected_action=result['action'],
        hold_minimum_m=min(hold), hold_endpoint_m=hold[-1],
        selected_minimum_m=minimum, selected_endpoint_m=endpoint,
        required_endpoint_gain_m=gain, unfinished_survey_target_preserved=True,
        nominal_footprint_and_full_reserve_unchanged=True)
    return result


class SurveyClearanceRepositionRuntime(FullReserveHeadingReleaseRuntime):
    def _select_clear_prediction(self, *args, **kwargs):
        result = super()._select_clear_prediction(*args, **kwargs)
        if self.terminal_position_approach:
            return result
        repositioned = reposition_survey(result)
        if repositioned is not result:
            self.clearance_turn = None
        return repositioned
