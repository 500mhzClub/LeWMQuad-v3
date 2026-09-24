"""Anticipate the unchanged dispatch stopping connector on observed map cells."""
import math
import numpy as np

from lewm.fine_stored_obstacle_routing_development import cached_clearance
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.stopping_margin_dispatch_development import STOPPING_ALLOWANCE_S, NOMINAL_RADIUS_M
from lewm.clearance_turn_recovery_development import choose


def stopping_projection_checks(prediction, cells, position, rotation, *, pulse=False):
    prediction = np.asarray(prediction, float)
    position, rotation = np.asarray(position, float), np.asarray(rotation, float)
    if (prediction.shape != (6, 8, 5) or position.shape != (3,) or rotation.shape != (3, 3)
            or not np.isfinite(prediction).all() or not np.isfinite(position).all()
            or not np.isfinite(rotation).all()):
        raise ValueError('finite saved candidate forecasts and observed map pose required')
    clearance = cached_clearance(cells)
    rows = []
    for i, action in enumerate(ACTIONS):
        command = np.asarray(candidate_commands(action)[0], float)
        if not any(command[:2]):
            rows.append(dict(action=action, translating=False, projection_clear=True, samples=[]))
            continue
        expiry_tick = 4 if pulse else 7
        samples = []
        # At dispatch (tick 3), the newest obstacle observation may be two
        # camera intervals old. Thereafter check each possible camera tick.
        # remaining time + observation age = expiry - observation time.
        for tick in range(1, expiry_tick):
            forecast = prediction[i, tick-1]
            yaw = math.atan2(forecast[2], forecast[3])
            c, s = math.cos(yaw), math.sin(yaw)
            turn = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            q = position + rotation @ np.r_[forecast[:2], 0.]
            Q = turn @ rotation
            horizon = (expiry_tick-tick)*.1 + STOPPING_ALLOWANCE_S
            endpoint = q + Q @ (command * np.array([horizon, horizon, 0.]))
            minimum = clearance.minimum(q[:2], endpoint[:2])
            samples.append(dict(observation_offset_ns=tick*100_000_000,
                requested_speed_horizon_s=horizon, minimum_clearance_m=minimum,
                projection_clear=minimum is None or minimum > NOMINAL_RADIUS_M+1e-12))
        rows.append(dict(action=action, translating=True,
            projection_clear=all(r['projection_clear'] for r in samples), samples=samples))
    return rows


def avoid_blocked_translation(selection, checks):
    """Preserve existing decisions unless their stopping projection is blocked."""
    from copy import deepcopy
    result = deepcopy(selection)
    selected = next(row for row in checks if row['action'] == selection['action'])
    result['planned_stopping_projection'] = dict(candidates=checks,
        before_action=selection['action'], changed=False,
        stopping_allowance_s=STOPPING_ALLOWANCE_S,
        anticipated_observation_age_maximum_s=.2,
        nominal_footprint_radius_m=NOMINAL_RADIUS_M,
        historical_obstacles_used=True, future_dispatch_obstacles_known=False,
        stopping_distance_bound_calibrated=False, dispatch_guards_unchanged=True)
    if not selected['translating'] or selected['projection_clear']:
        return result
    utilities = {r['action']:r['utility_m'] for r in selection.get('scan_utilities', selection['candidates'])}
    # Do not introduce another translation after a higher-level route or
    # arrival decision has already selected one. Continue aligning, or hold.
    eligible = [i for i,row in enumerate(selection['memory_forecast_candidates'])
        if row['action'] in ('hold', 'left_turn', 'right_turn')
        and row['nominal_predicted_path_clear'] and row['action'] in utilities]
    index = max(eligible, key=lambda i:utilities[ACTIONS[i]]) if eligible else ACTIONS.index('hold')
    choose(result, index)
    result['planned_stopping_projection'].update(changed=True, after_action=result['action'])
    return result


class PlannedStoppingProjectionMixin:
    def _select_clear_prediction(self, selected, prediction, snapshot, position, rotation):
        result = super()._select_clear_prediction(selected, prediction, snapshot, position, rotation)
        checks = stopping_projection_checks(prediction, snapshot.fine_occupied, position, rotation,
            pulse=bool(self.planning_translation_pulse))
        revised = avoid_blocked_translation(result, checks)
        if revised['planned_stopping_projection']['changed']:
            self.clearance_turn = None
        return revised
