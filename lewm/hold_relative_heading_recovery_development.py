"""Test heading progress during a hold deadlock without losing hold clearance."""
from copy import deepcopy
import math

from lewm.clearance_turn_recovery_development import choose
from lewm.geometry_progress_pilot_development import ACTIONS


def recover_heading_against_hold(selection):
    preferred = selection.get('before_memory_filter_action')
    rows = selection.get('memory_forecast_candidates', [])
    if (selection.get('action') != 'hold' or preferred not in ('left_turn', 'right_turn')
            or [r['action'] for r in rows] != list(ACTIONS)
            or any(r['nominal_predicted_path_clear'] for r in rows if r['action'] != 'hold')):
        return selection
    by_action = {r['action']: r for r in rows}
    hold = by_action['hold']['segment_clearances_m']
    turn = by_action[preferred]['segment_clearances_m']
    if (len(hold) != 8 or len(turn) != 8
            or not all(v is not None and math.isfinite(v) for v in hold+turn)
            or min(hold) <= .45 or min(turn) < min(hold) or turn[-1] < hold[-1]):
        return selection
    utilities = {r['action']: r['utility_m'] for r in selection.get('scan_utilities', selection['candidates'])}
    if (not all(a in utilities and math.isfinite(utilities[a]) for a in ('hold', preferred))
            or utilities[preferred] <= utilities['hold']):
        return selection
    if 'scan_utilities' not in selection:
        candidate = next(r for r in selection['candidates'] if r['action'] == preferred)
        start = candidate['predicted_heading_error_at_commit_start_rad']
        end = candidate['predicted_heading_error_at_commit_end_rad']
        if not math.isfinite(start) or not math.isfinite(end) or end >= start:
            return selection
    result = deepcopy(selection)
    index = ACTIONS.index(preferred)
    result['memory_forecast_candidates'][index].update(nominal_predicted_path_clear=True,
        reserve_recovery_path_clear=True,
        clearance_check_mode='HEADING_PROGRESS_WITHIN_HOLD_RELATIVE_MINIMUM')
    result.update(memory_forecast_status='HOLD_RELATIVE_HEADING_RECOVERY',
        recovery_restores_reserve_by_commit_end=False,
        clearance_turn=dict(active=False, event='HOLD_RELATIVE_HEADING_RESELECTS_EACH_PLAN'),
        hold_relative_heading_recovery=dict(applied=True, selected_action=preferred,
            hold_minimum_m=min(hold), hold_endpoint_m=hold[-1],
            selected_minimum_m=min(turn), selected_endpoint_m=turn[-1],
            previous_utility_m=utilities['hold'], selected_utility_m=utilities[preferred],
            nominal_footprint_radius_m=.45, minimum_clearance_gain_required=False,
            full_reserve_restoration_claimed=False,
            predicted_clearance_is_not_execution_certificate=True))
    return choose(result, index)


class HoldRelativeHeadingRecoveryMixin:
    def _select_clear_prediction(self, *args, **kwargs):
        result = super()._select_clear_prediction(*args, **kwargs)
        if self.terminal_position_approach:
            return result
        recovered = recover_heading_against_hold(result)
        if recovered is not result:
            self.clearance_turn = None
        return recovered
