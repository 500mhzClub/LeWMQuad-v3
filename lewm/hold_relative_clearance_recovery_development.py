"""Escape a reserve deadlock when a measured-map forecast improves on hold."""
from copy import deepcopy
import math
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.clearance_turn_recovery_development import choose
from lewm.terminal_translation_pulse_development import TerminalTranslationPulseRuntime


def recover_against_hold(selection):
    rows = selection.get('memory_forecast_candidates', [])
    if (selection.get('action') != 'hold' or len(rows) != 6
            or any(row['nominal_predicted_path_clear'] for row in rows if row['action'] != 'hold')):
        return selection
    hold = next(row for row in rows if row['action'] == 'hold')['segment_clearances_m']
    def finite_path(values):
        return len(values) == 8 and all(v is not None and math.isfinite(v) for v in values)
    if not finite_path(hold) or min(hold) <= .45:
        return selection
    eligible = []
    for i, row in enumerate(rows):
        if row['action'] == 'hold':
            continue
        path = row['segment_clearances_m']
        gain = max(.001, .1 * max(0., row['required_path_clearance_m'] - min(hold)))
        if (finite_path(path) and min(path) > .45 and min(path) >= min(hold)
                and path[-1] >= hold[-1] + gain):
            eligible.append((min(path), path[-1], i, gain))
    if not eligible:
        return selection
    minimum, endpoint, index, gain = max(eligible)
    result = deepcopy(selection)
    row = result['memory_forecast_candidates'][index]
    row.update(nominal_predicted_path_clear=True, reserve_recovery_path_clear=True,
        clearance_check_mode='IMPROVES_ON_HOLD_WITHIN_NOMINAL_FOOTPRINT')
    result.update(memory_forecast_status='HOLD_RELATIVE_CLEARANCE_RECOVERY',
        recovery_restores_reserve_by_commit_end=False,
        hold_relative_clearance_recovery=dict(selected=True,
            hold_minimum_m=min(hold), hold_endpoint_m=hold[-1],
            selected_minimum_m=minimum, selected_endpoint_m=endpoint,
            required_endpoint_gain_m=gain, nominal_footprint_radius_m=.45,
            predicted_clearance_is_not_execution_certificate=True),
        clearance_turn=dict(active=False, event='HOLD_RELATIVE_RECOVERY_RESELECTS_EACH_PLAN'))
    return choose(result, ACTIONS.index(row['action']))


class HoldRelativeClearanceRecoveryRuntime(TerminalTranslationPulseRuntime):
    def _select_clear_prediction(self, selected, prediction, snapshot, position, rotation):
        original = super()._select_clear_prediction(selected, prediction, snapshot, position, rotation)
        result = recover_against_hold(original)
        if result is not original:
            self.clearance_turn = None
        return result
