"""Development fixes for the shared-harness traps (development mode, 30 Sep 2026).

Each fix is a mixin placed ahead of the frozen V4 `CompletedSupportRuntimeMixin`. No frozen file
is edited; the development owner swaps the composed mixin in through `bind`.

- `terminal`: near the goal, score positional progress only. The frozen scorer adds a
  heading-alignment term scaled by min(0.35 m, distance). Within a few centimetres it
  dominates, and in-place turns win even though they never bring an in-footprint target round
  (the terminal limit cycle; for example C1 validation 10/0 chose 99 right turns in a row, 4.5 cm
  from home). The arrival definition (2 cm observed, 4 cm physical, 1-s dwell) is unchanged.
- `latch`: time out the clearance-turn recovery latch. The frozen latch releases only when the
  measured heading reaches its target. When the latched direction is blocked by forecast
  clearance it holds forever (C1 val 13, C3 val 17, C3-v2 fresh 01 and 06), or it flips direction
  repeatedly (C4 val 22). If the latch makes no heading progress (at least 0.05 rad) over
  LATCH_STALL_DECISIONS decisions, or flips direction LATCH_MAX_SWITCHES times, it is
  released. The best clearance-feasible candidate is then chosen without it, exactly as the
  frozen reserve logic ranks them, and re-latching is blocked for LATCH_COOLDOWN decisions.
- `deadlock`: escape the no-eligible-movement hold. The deadlocks are hair-trigger: the robot
  sits a few millimetres inside the 0.45-m nominal disk (for example 0.446 m in C3 val 12), so
  every candidate, hold included, forecasts below the requirement. Translations are
  view-restricted and turns are clearance-blocked, so it holds forever.
  After ESCAPE_AFTER consecutive such holds:
  - selection picks a view-permitted in-place turn whose predicted minimum path clearance is at
    least max(ESCAPE_FLOOR_M, current - ESCAPE_SLACK_M), preferring the waypoint side;
  - while the escape is armed, dispatch allows pure turns if the current observed disk of
    radius ESCAPE_FLOOR_M is clear. The frozen check uses 0.45 m; 0.42 m still clears the Go2's
    turning sweep of about 0.39 m.
  Translations are never relaxed, and unobserved floor is never entered.
"""
import math

import numpy as np

from lewm.clearance_turn_recovery_development import choose, wrap
from lewm.observed_geometry_refinement_development import nominal_connector
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.mission_coordinate_metric_development import position_distance

TERMINAL_RADIUS_M = .10


class TerminalPositionScoringMixin:
    """Within TERMINAL_RADIUS_M of the scored waypoint, select by position/contact utility alone."""

    def _score(self, prediction, goal_body, **kwargs):
        result = super()._score(prediction, goal_body, **kwargs)
        distance = float(position_distance(goal_body, kwargs.get('position_metric_matrix')))
        if distance >= TERMINAL_RADIUS_M or not result.get('candidates') or 'position_contact_utility_m' not in result['candidates'][0]:
            return result
        for row in result['candidates']:
            row['alignment_dropped_terminal'] = True
            row['utility_m'] = row['position_contact_utility_m']
        selected = max(range(len(result['candidates'])), key=lambda i: result['candidates'][i]['utility_m'])
        action = result['candidates'][selected]['action']
        result.update(action=action, action_index=selected, requested_command=candidate_commands(action)[0],
                      selection_objective='terminal_position_progress_minus_contact', terminal_alignment_dropped=True,
                      terminal_radius_m=TERMINAL_RADIUS_M)
        return result


LATCH_STALL_DECISIONS, LATCH_MAX_SWITCHES, LATCH_COOLDOWN = 10, 3, 15


def unlatched_choice(result, event):
    """The frozen reserve ranking without the latch: best nominal-clear candidate, else hold."""
    utilities = {r['action']: r['utility_m'] for r in result.get('scan_utilities', result['candidates'])}
    eligible = [i for i, r in enumerate(result['memory_forecast_candidates'])
                if r['action'] in utilities and r['nominal_predicted_path_clear']]
    index = max(eligible, key=lambda i: utilities[ACTIONS[i]]) if eligible else ACTIONS.index('hold')
    choose(result, index)
    result['clearance_turn'] = dict(active=False, event=event)
    return result


class LatchTimeoutMixin:
    def __init__(self, *args, **kwargs):
        self._latch_track, self._latch_cooldown = None, 0
        super().__init__(*args, **kwargs)

    def _select_clear_prediction(self, selected, prediction, snapshot, position, rotation):
        result = super()._select_clear_prediction(selected, prediction, snapshot, position, rotation)
        latch = self.clearance_turn
        active = (result.get('clearance_turn') or {}).get('active')
        if self._latch_cooldown > 0:
            self._latch_cooldown -= 1
            if active and 'memory_forecast_candidates' in result:
                self.clearance_turn, self._latch_track = None, None
                result = unlatched_choice(result, 'DEV_LATCH_SUPPRESSED_COOLDOWN')
            return result
        if latch is not None and active and 'memory_forecast_candidates' in result:
            heading = math.atan2(rotation[1, 0], rotation[0, 0])
            remaining = abs(wrap(latch['target_heading_rad']-heading))
            key = (latch['target_heading_rad'], latch['mission_generation'])
            track = self._latch_track
            if track is None or track['key'] != key:
                track = dict(key=key, best=remaining, stalled=0, switches=latch.get('reserve_recovery_direction_switches', 0))
            if remaining < track['best']-.05:
                track.update(best=remaining, stalled=0)
            else:
                track['stalled'] += 1
            switches = latch.get('reserve_recovery_direction_switches', 0)-track['switches']
            if track['stalled'] >= LATCH_STALL_DECISIONS or switches >= LATCH_MAX_SWITCHES:
                self.clearance_turn, self._latch_track, self._latch_cooldown = None, None, LATCH_COOLDOWN
                result = unlatched_choice(result, 'DEV_LATCH_TIMEOUT_RELEASED')
                result['dev_latch_timeout'] = dict(stalled_decisions=track['stalled'], direction_switches=switches)
            else:
                self._latch_track = track
        return result


ESCAPE_AFTER, ESCAPE_FLOOR_M, ESCAPE_SLACK_M = 5, .42, .003


class DeadlockEscapeMixin:
    def __init__(self, *args, **kwargs):
        self._deadlock_holds, self._escape_armed = 0, False
        super().__init__(*args, **kwargs)

    def _select_clear_prediction(self, selected, prediction, snapshot, position, rotation):
        result = super()._select_clear_prediction(selected, prediction, snapshot, position, rotation)
        rows = result.get('memory_forecast_candidates')
        if not rows or result['action'] != 'hold':
            self._deadlock_holds, self._escape_armed = 0, False
            return result
        utilities = {r['action']: r['utility_m'] for r in result.get('scan_utilities', result['candidates'])}
        movable = [r for r in rows if r['action'] != 'hold' and r['action'] in utilities and r['nominal_predicted_path_clear']]
        if movable or (result.get('clearance_turn') or {}).get('active'):
            self._deadlock_holds, self._escape_armed = 0, False
            return result
        self._deadlock_holds += 1
        if self._deadlock_holds < ESCAPE_AFTER:
            return result
        current = next(r['minimum_predicted_path_clearance_m'] for r in rows if r['action'] == 'hold')
        floor = max(ESCAPE_FLOOR_M, (current or ESCAPE_FLOOR_M)-ESCAPE_SLACK_M)
        options = [i for i, r in enumerate(rows) if r['action'] in ('left_turn', 'right_turn') and r['action'] in utilities
                   and r['minimum_predicted_path_clearance_m'] is not None and r['minimum_predicted_path_clearance_m'] >= floor]
        if not options:
            return result
        goal = result.get('waypoint_body_xy_m')
        side = 'left_turn' if goal is None or math.atan2(goal[1], goal[0]) > 0 else 'right_turn'
        index = max(options, key=lambda i: (rows[i]['action'] == side, rows[i]['minimum_predicted_path_clearance_m']))
        choose(result, index)
        self._escape_armed = True
        result['dev_deadlock_escape'] = dict(consecutive_no_eligible_holds=self._deadlock_holds, current_clearance_m=current,
                                             clearance_floor_m=floor, action=rows[index]['action'], dispatch_disk_radius_m=ESCAPE_FLOOR_M)
        return result

    def request(self, *, now_ns):
        result = super().request(now_ns=now_ns)
        if not self._escape_armed or result['reason'] not in ('CURRENT_OBSERVED_OBSTACLE_VETO', 'COMMAND_WINDOW_VETO_LATCHED'):
            return result
        with self.lock:
            live = [q for q in self.plans if q.dispatch_ns <= now_ns < q.expires_ns]
            plan, current = (live[-1] if live else None), self.latest_obstacles
        if plan is None or current is None or any(plan.command[:2]) or not plan.command[2]:
            return result
        p = np.asarray(current.position_map)
        check = nominal_connector(p[:2], p[:2], sorted(current.occupied), radius_m=ESCAPE_FLOOR_M)
        if not check['nominal_disk_connector_clear']:
            return result
        with self.lock:
            self.rejected_windows.pop(plan.observed_ns, None)
        return result | dict(requested_command=plan.request(now_ns=now_ns, fresh_observation_allows_motion=True),
                             reason='CURRENT_NOMINAL_OBSTACLE_TEST_PASSED', dev_escape_turn_dispatch=dict(
                                 frozen_reason=result['reason'], disk_radius_m=ESCAPE_FLOOR_M, connector=check))


FIXES = {'terminal': TerminalPositionScoringMixin, 'latch': LatchTimeoutMixin, 'deadlock': DeadlockEscapeMixin}


def compose(fixes, base):
    """The frozen startup mixin with the requested fixes ahead of it."""
    unknown = set(fixes)-set(FIXES)
    if unknown:
        raise ValueError(f'unknown fixes: {sorted(unknown)}')
    bases = tuple(FIXES[f] for f in sorted(fixes))+(base,)
    return type('DevStartupRecoveryRuntimeMixin', bases, {})
