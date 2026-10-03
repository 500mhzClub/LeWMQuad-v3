"""Next harness version, development: reserve exit and C2's aligned clearance (Andrew, 2-3 October 2026).

Two changes to the frozen V4 harness and nothing else (docs/go2_navigation_harness_reserve_exit_plan_2026-10-02.md):

(a) Reserve exit, for every forecast-based clearance check. Under every forecast controller (C0, C1, C3, C4) the
    innermost check is clearance_lookahead_development.ReserveRecoveryLookaheadRuntime calling
    memory_forecast_clearance_development.select_clear_prediction. It now also passes a translation whose forecast
    centre-path clearance to remembered walls never decreases along the path (1-mm tolerance, step to step and against
    the start) and ends higher than it starts, even when it starts inside the 0.48-m reserve or the 0.45-m disc.
    Everything else about the check is unchanged, and every outer layer (turn reserve and recovery, clearance-turn
    latch, stopping projection, terminal rules) sees such a move as an ordinary clear candidate.
(b) C2's clearance semantics match the others. C2's reactive rule no longer makes every action ineligible within
    0.45 m. Each candidate is checked on its nominal command path, with the same check as (a): the committed prefix,
    the candidate's 400-ms commit and one stopped tick, integrated at 10 ms with unicycle kinematics (no learned model,
    no command-history fit). That includes the turn reserve with stepwise recovery, as the forecast controllers run it,
    and the exit. C2's selection rule is unchanged: the nearest primitive to the waypoint feedback among eligible
    actions. This gives C2 a crude kinematic forecast for its safety check; its selection uses no forecast.

Both changes act only inside the mixins below (context variables). Importing this module rebinds two module-level
names, but with the context variables unset the rebound functions return exactly the frozen results. Missions use it
through scripts/run_go2_dev_mission_pinned_v3_development.py.
"""
import contextvars
import math

import numpy as np

from lewm import clearance_lookahead_development as _lookahead
from lewm import persistent_visual_baselines_development as _c2
from lewm.clearance_turn_recovery_development import reserve_turns
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.memory_forecast_clearance_development import select_clear_prediction as _frozen_select

HARNESS = 'reserve_exit_v1'
EXIT_TOLERANCE_M = .001
TRANSLATIONS = ('forward', 'left_arc', 'right_arc')
TICK_S, SUBSTEPS, COMMIT_TICKS, HORIZON_TICKS = .1, 10, 4, 8
_EXIT = contextvars.ContextVar('dev_reserve_exit', default=False)
_C2_PREFIX = contextvars.ContextVar('dev_c2_prefix', default=None)
_C2_CHECK = contextvars.ContextVar('dev_c2_nominal_path_check', default=None)
_frozen_reactive = _c2.select_reactive


def exit_clear(distances, tolerance=EXIT_TOLERANCE_M):
    """The forecast centre path never loses clearance (within tolerance) and ends with more than it starts with."""
    if len(distances) != HORIZON_TICKS or any(d is None for d in distances):
        return False
    d = [float(v) for v in distances]
    return bool(all(b >= a-tolerance for a, b in zip(d, d[1:])) and min(d) >= d[0]-tolerance and d[-1] > d[0])


def apply_exit(result):
    """Pass blocked translations that leave the reserve; reselect with the frozen check's own rule and utilities."""
    if not result.get('reserve_recovery_enabled'):
        raise ValueError('reserve exit requires the reserve-recovery check (segment clearances)')
    rows = result['memory_forecast_candidates']
    exits = []
    for row in rows:
        passes = bool(row['action'] in TRANSLATIONS and not row['nominal_predicted_path_clear']
                      and exit_clear(row['segment_clearances_m']))
        row['reserve_exit_path_clear'] = passes
        if passes:
            row.update(nominal_predicted_path_clear=True, clearance_check_mode='RESERVE_EXIT')
            exits.append(row['action'])
    utilities = {r['action']: r['utility_m'] for r in result.get('scan_utilities', result['candidates'])}
    eligible = [i for i, r in enumerate(rows) if r['action'] in utilities and r['nominal_predicted_path_clear']]
    index = max(eligible, key=lambda i: utilities[ACTIONS[i]]) if eligible else ACTIONS.index('hold')
    action = ACTIONS[index]
    result.update(action=action, action_index=index, requested_command=candidate_commands(action)[0],
                  memory_forecast_status='CLEAR_CANDIDATE_SELECTED' if eligible else 'NO_CLEAR_CANDIDATE_ZERO_REQUESTED',
                  selected_reserve_recovery=bool(rows[index]['reserve_recovery_path_clear']),
                  reserve_exit_enabled=True, reserve_exit_candidates=exits,
                  selected_reserve_exit=bool(rows[index]['reserve_exit_path_clear']),
                  reserve_exit_rule=dict(harness=HARNESS, tolerance_m=EXIT_TOLERANCE_M,
                                         rule='translation; centre-path clearance never decreases and ends higher than it starts'))
    return result


def _select_with_exit(*args, **kwargs):
    result = _frozen_select(*args, **kwargs)
    return apply_exit(result) if _EXIT.get() else result


def nominal_prediction(prefix):
    """(6, 8, 3) body-frame x, y, yaw at 0.1-0.8 s: committed prefix, the candidate's 400-ms commit, one stopped tick."""
    prefix = np.asarray(prefix, float).reshape(-1, 3)
    if len(prefix)+COMMIT_TICKS+1 != HORIZON_TICKS or not np.isfinite(prefix).all():
        raise ValueError('three committed prefix commands required')
    dt = TICK_S/SUBSTEPS
    out = np.zeros((len(ACTIONS), HORIZON_TICKS, 3))
    for i, action in enumerate(ACTIONS):
        commit = candidate_commands(action)
        if len(commit) != COMMIT_TICKS*SUBSTEPS:
            raise ValueError('400-ms candidate at 10-ms resolution required')
        commands = [c for c in prefix for _ in range(SUBSTEPS)]+[np.asarray(c, float) for c in commit]+[np.zeros(3)]*SUBSTEPS
        x = y = yaw = 0.
        for k, (vx, vy, wz) in enumerate(commands, 1):
            mid = yaw+.5*wz*dt
            x += (vx*math.cos(mid)-vy*math.sin(mid))*dt
            y += (vx*math.sin(mid)+vy*math.cos(mid))*dt
            yaw += wz*dt
            if k % SUBSTEPS == 0:
                out[i, k//SUBSTEPS-1] = (x, y, yaw)
    return out


def nominal_path_rows(prefix, cells, position, rotation):
    """The forecast controllers' check (translation reserve, reserve recovery, exit, stepwise turn reserve) on nominal paths."""
    neutral = dict(action='hold', candidates=[dict(action=a, utility_m=0.) for a in ACTIONS])
    result = _frozen_select(neutral, nominal_prediction(prefix), cells, position, rotation,
                            translation_reserve_m=.03, reserve_recovery=True)
    return reserve_turns(apply_exit(result), stepwise=True)['memory_forecast_candidates']


def _reactive_with_nominal_check(goal_body_xy, *, scan_error=None, current_clearance_m=None):
    context = _C2_CHECK.get()
    if context is None:
        return _frozen_reactive(goal_body_xy, scan_error=scan_error, current_clearance_m=current_clearance_m)
    result = _frozen_reactive(goal_body_xy, scan_error=scan_error, current_clearance_m=None)  # C2's own rule, no all-actions block
    rows = nominal_path_rows(context['prefix'], context['cells'], context['position'], context['rotation'])
    checked = {r['action']: r for r in rows}
    for row in result['candidates']:
        clear = bool(checked[row['action']]['nominal_predicted_path_clear'])
        row.update(rule_eligible=row['eligible'], nominal_path_clear=clear, eligible=bool(row['eligible'] and clear))
    eligible = [i for i, row in enumerate(result['candidates']) if row['eligible']]
    index = (min(eligible, key=lambda i: result['candidates'][i]['normalized_command_distance_squared'])
             if eligible else ACTIONS.index('hold'))
    action = ACTIONS[index]
    result.update(action=action, action_index=index, requested_command=candidate_commands(action)[0],
                  current_stored_clearance_m=current_clearance_m,
                  current_nominal_disk_clear=current_clearance_m is None or current_clearance_m > .45+1e-12,
                  all_actions_block_removed=True, predicted_feasibility_used=True,
                  rule='nearest_six_action_primitive_to_current_waypoint_feedback_among_nominal_path_clear',
                  c2_nominal_path_check=dict(harness=HARNESS, rows=rows, selected_reserve_exit=bool(checked[action].get('reserve_exit_path_clear')),
                                             path='committed prefix, 400-ms commit, one stopped tick; unicycle integration at 10 ms; no learned model'))
    return result


_lookahead.select_clear_prediction = _select_with_exit
_c2.select_reactive = _reactive_with_nominal_check


class ReserveExitMixin:
    """(a): the reserve exit inside every forecast-based clearance check."""
    dev_harness_version = HARNESS

    def _select_clear_prediction(self, *args, **kwargs):
        token = _EXIT.set(True)
        try:
            return super()._select_clear_prediction(*args, **kwargs)
        finally:
            _EXIT.reset(token)


class C2NominalPathCheckMixin:
    """(b): C2's eligibility from the same check on nominal command paths, instead of its all-actions block."""

    def _select_action(self, *args, **kwargs):
        prefix = kwargs['prefix'] if 'prefix' in kwargs else args[2]
        token = _C2_PREFIX.set(prefix)
        try:
            return super()._select_action(*args, **kwargs)
        finally:
            _C2_PREFIX.reset(token)

    def _select_clear_prediction(self, selected, prediction, snapshot, position, rotation):
        prefix = _C2_PREFIX.get()
        if prefix is None:
            raise RuntimeError('C2 nominal-path check needs the committed prefix from _select_action')
        token = _C2_CHECK.set(dict(prefix=prefix, cells=snapshot.fine_occupied, position=position, rotation=rotation))
        try:
            return super()._select_clear_prediction(selected, prediction, snapshot, position, rotation)
        finally:
            _C2_CHECK.reset(token)


def mixins_for(arm):
    """Outermost mixins for this harness version."""
    return (ReserveExitMixin,)+((C2NominalPathCheckMixin,) if arm == 'C2' else ())
