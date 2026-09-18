"""Use an already-clear arc when measured-view recovery cannot turn in place."""
from copy import deepcopy
import math

import numpy as np

from lewm.clearance_turn_recovery_development import choose, wrap
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.visual_support_recovery_development import MAX_VIEW_DISTANCE_M


def recover_view_with_arc(selection, prediction, reference_body_m):
    """Apply only after the ordinary clearance and stopping filters have run."""
    if selection['action'] != 'hold' or 'scan_utilities' not in selection:
        return selection
    rows = {r['action']: r for r in selection['memory_forecast_candidates']}
    if any(rows[a]['nominal_predicted_path_clear'] for a in ('left_turn', 'right_turn')):
        return selection
    if not rows['hold']['nominal_footprint_path_clear']:
        return selection
    checks = {r['action']: r for r in selection['planned_stopping_projection']['candidates']}
    prediction = np.asarray(prediction, float)
    reference = np.asarray(reference_body_m, float)
    if (prediction.shape != (6, 8, 5) or reference.shape != (3,)
            or not np.isfinite(prediction).all() or not np.isfinite(reference).all()):
        raise ValueError('finite complete forecasts and measured local view required')
    error = selection['scan_heading_error_rad']
    if not math.isfinite(error):
        raise ValueError('finite measured heading error required')
    if np.linalg.norm(reference) > MAX_VIEW_DISTANCE_M or abs(error) <= .1:
        return selection

    def heading_gain(index):
        yaw = math.atan2(prediction[index, 6, 2], prediction[index, 6, 3])
        yaw -= math.atan2(prediction[index, 2, 2], prediction[index, 2, 3])
        return abs(error)-abs(wrap(error-yaw))

    hold_gain = heading_gain(ACTIONS.index('hold'))
    eligible = []
    for action in ('left_arc', 'right_arc'):
        i = ACTIONS.index(action)
        row = rows[action]
        if not (row['nominal_predicted_path_clear'] and row['nominal_footprint_path_clear']
                and row['clearance_check_mode'] in ('FULL_RESERVE', 'RESERVE_RECOVERY')
                and checks[action]['projection_clear']):
            continue
        if candidate_commands(action)[0][2]*error <= 0:
            continue
        gain = heading_gain(i)
        path = np.column_stack((prediction[i, :, :2], np.zeros(8)))
        max_distance = float(np.linalg.norm(path-reference, axis=1).max())
        if gain <= max(0., hold_gain) or max_distance > MAX_VIEW_DISTANCE_M:
            continue
        eligible.append((gain, -i, max_distance))
    if not eligible:
        return selection
    gain, negative_index, max_distance = max(eligible)
    result = deepcopy(selection)
    choose(result, -negative_index)
    result['measured_view_arc_recovery'] = dict(applied=True,
        previous_action='hold', selected_action=result['action'],
        predicted_heading_gain_rad=gain, hold_heading_gain_rad=hold_gain,
        maximum_predicted_distance_from_reference_m=max_distance,
        reference_distance_limit_m=MAX_VIEW_DISTANCE_M,
        clearance_mode=rows[result['action']]['clearance_check_mode'],
        stopping_projection_clear=True, original_view_target_preserved=True,
        nominal_footprint_and_reserve_rules_unchanged=True,
        measured_dispatch_guards_unchanged=True, execution_safety_certified=False)
    return result


class MeasuredViewArcRecoveryMixin:
    def _route(self, snapshot, evidence, goal, *, measured_ns):
        result = super()._route(snapshot, evidence, goal, measured_ns=measured_ns)
        self.measured_view_arc_reference = None
        if result['status'] == 'LOW_VISUAL_SUPPORT_REQUIRES_MEASURED_VIEW':
            receipt = evidence['visual_support']
            active = receipt.get('recovery_state_at_observation')
            if active is not None:
                p, rotation, pose = self._pose(evidence, identity=(0, 0, 0), now_ns=measured_ns)
                if receipt['frame'] != pose['frame'] or receipt['measured_ns'] != measured_ns:
                    raise ValueError('view recovery must use its own measured planning pose')
                self.measured_view_arc_reference = np.asarray(rotation).T @ (
                    np.asarray(active['position'])-np.asarray(p))
        return result

    def _select_clear_prediction(self, selected, prediction, snapshot, position, rotation):
        result = super()._select_clear_prediction(selected, prediction, snapshot, position, rotation)
        reference = getattr(self, 'measured_view_arc_reference', None)
        if reference is None:
            return result
        revised = recover_view_with_arc(result, prediction, reference)
        if revised is not result:
            self.clearance_turn = None
        return revised
