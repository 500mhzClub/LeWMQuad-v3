"""Development fixes for the shared-harness traps (development mode, 30 Sep 2026).

Each fix is a mixin placed ahead of the frozen V4 `CompletedSupportRuntimeMixin`. No frozen file
is edited; the development owner swaps the composed mixin in through `bind`.

- `terminal`: break the terminal limit cycle. The frozen scorer adds a heading-alignment term
  scaled by min(0.35 m, distance). Within a few centimetres it can dominate, so in-place turns
  win even though they never bring an in-footprint target round (C1 validation 10/0 chose 99
  right turns in a row, 4.5 cm from home). Within TERMINAL_RADIUS_M, after
  TERMINAL_SPIN_TURNS consecutive turn selections, the next TERMINAL_BURST_DECISIONS
  decisions are selected by position/contact utility alone; then the frozen scorer resumes.
  (The first version dropped the alignment term throughout the 0.10-m radius. That regressed
  C1's home approach on fresh-check 03, 04 and 08: about 800 decisions hovering 6-11 cm
  from home, where the originals arrived at 1 cm.) The arrival definition (2 cm observed,
  4 cm physical, 1-s dwell) is unchanged.
  C2's reactive terminal rule (turn to face the target within 0.1 rad, then pulse forward)
  has its own limit cycle: on validation 14, 27 and 29 it turns the same way for ~1,000
  decisions 2-5 cm from the goal, because the target sits near the Go2's turning centre and
  turning in place carries it round. After TERMINAL_SPIN_TURNS consecutive same-direction
  terminal turns, one existing 100-ms forward pulse is taken instead, which moves the robot
  off that point; the heading-first rule then resumes.
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
  C2's reactive selector deadlocks the same way, only more often: 6 of its 11 validation
  failures end in about 1,000 consecutive holds at a stored clearance of 0.41-0.44 m. It
  gates every action on the current stored clearance exceeding 0.45 m, so in-place turns
  cannot free it. After ESCAPE_AFTER such holds, provided the clearance is at least
  REACTIVE_ESCAPE_FLOOR_M:
  - selection picks the forward/arc move closest to C2's own desired command whose 400-ms
    requested path keeps the stored-map clearance at least (current - ESCAPE_SLACK_M) and ends
    at least REACTIVE_ESCAPE_GAIN_M further from stored obstacles; if none exists, it turns
    toward the heading with the most clearance 0.1 m away;
  - dispatch allows that move if the currently observed connector clears
    REACTIVE_ESCAPE_FLOOR_M and does not end closer to an observed obstacle than it starts.
- `stall`: retire a frontier the robot cannot make progress toward. C1 fresh-check 09 spends
  160+ s at one spot, turning back and forth toward a frontier route point 0.38 m away that
  its clearance-limited turns never face. Latch releases do not help: it re-latches, and it
  never holds, so `deadlock` does not fire. On the outbound leg, while routing to a frontier,
  if the robot moves less than STALL_RADIUS_M over STALL_NS, the active camera-frontier visit
  is abandoned (its viewpoint recorded as tried) and every floor cell within
  STALL_EXCLUSION_RADIUS_M of the abandoned frontier (the visit's target cell, else the route
  end) is excluded from target selection for STALL_EXCLUSION_NS. The radius covers the whole
  unknown pocket: at C1 fresh-check 09, retiring 0.3 m around the viewpoint only moved the
  choice to the neighbouring frontier cell, served from the same viewpoint, and control was
  bit-identical. The exclusion survives the visit logic's clear-on-new-map rule.
  Exclusion only changes which frontier is selected; no cell is marked free or blocked. An
  exclusion that leaves neither a frontier nor a goal route is undone at once (C1 validation
  13: excluding the only frontier left the robot spinning for the 90-s exclusion, and the
  mission ran out of budget; without `stall` it succeeds).
- `backup`: a scripted short back-up inside the escape rules (Andrew, 30 September evening).
  Reverse is not added to the candidate bank, because no predictor is trained on it. The
  deadlock escape requests it when no turn or move qualifies; the stall watchdog requests it
  on the first stall at a spot (within BACKUP_REPEAT_RADIUS_M), before retiring frontiers.
  The script requests BACKUP_SPEED_MPS in reverse for BACKUP_DECISIONS planning cycles
  (0.20 m). The selection records `hold`, so no forecast is attributed to it, and the stored
  plan carries the reverse command. Every step requires:
  - the 400-ms reverse segment to lie on observed stored floor cells;
  - stored-map clearance along it of at least BACKUP_FLOOR_M, and no more than 1 cm below the
    current clearance;
  - at dispatch, the currently observed connector to clear BACKUP_FLOOR_M without ending
    closer to an observed obstacle.
  Any failed check ends the script. The predictors then see a reverse command in their
  committed prefix for a few decisions, which is outside their training.
- `coverage` (harness fix, 2 October; on in both recovery settings): the frozen
  translation-footprint coverage rule (`lewm/coverage_translation_view_development.py`) counts
  every coarse cell that is not floor as unknown, including cells already observed as occupied
  (obstacle edges the fine clearance gate passes). Its remedy, a camera view request, targets
  only unobserved cells, so the robot held until its utilities drifted (C3 maze 30: 24.8 s;
  1-3% of mission time in the preliminary run). With this fix, observed occupied cells count as
  observed: only truly unobserved cells trigger the rule, and those also get a view request.
  Occupied cells stay with the clearance gate. Implemented without editing the frozen file: the
  module's `filter_translation` is wrapped once, and the wrapper changes behaviour only inside
  a runtime that carries this mixin (a context variable).
- `pose`: record why visual pose was lost. The tracker is terminal after its first failure,
  and every downstream consumer (registration, map, routing) re-derives the pose from the
  tracker's own evidence chain, so re-anchoring means re-plumbing those validators. The one
  pose loss seen on this harness (C1 fresh-check 09) came at the end of a flipping latched
  turn, facing a featureless wall, so `latch` targets its cause. This fix prints the
  tracker's failure chain to worker.log and then faults exactly as before.
"""
import contextvars
from dataclasses import replace
import json
import math

import numpy as np

from lewm.clearance_turn_recovery_development import choose, wrap
from lewm.fine_stored_obstacle_routing_development import cached_clearance
from lewm.observed_geometry_refinement_development import nominal_connector, segment_cell_distances
from lewm.observed_floor_waypoint_development import segment_cells
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.mission_coordinate_metric_development import position_distance
from lewm.process_mapped_runtime_development import pose_update
from lewm import coverage_translation_view_development as coverage_rule

TERMINAL_RADIUS_M = .10


class TerminalPositionScoringMixin:
    """After a terminal turn spin near the waypoint, briefly select by position/contact utility alone."""

    def _score(self, prediction, goal_body, **kwargs):
        result = super()._score(prediction, goal_body, **kwargs)
        distance = float(position_distance(goal_body, kwargs.get('position_metric_matrix')))
        # A zero goal is the planner's scan mode (panorama or view request); the scan selector
        # overrides this choice, so it is not a terminal approach.
        scanning = not np.any(np.asarray(goal_body, float))
        if scanning or distance >= TERMINAL_RADIUS_M or not result.get('candidates') or 'position_contact_utility_m' not in result['candidates'][0]:
            self._terminal_turns, self._terminal_burst = 0, 0
            return result
        if self._terminal_burst == 0:
            self._terminal_turns = self._terminal_turns+1 if result['action'] in ('left_turn', 'right_turn') else 0
            if self._terminal_turns < TERMINAL_SPIN_TURNS:
                return result
            self._terminal_burst, self._terminal_turns = TERMINAL_BURST_DECISIONS, 0
            result['dev_terminal_spin_break'] = dict(kind='position_scoring_burst', consecutive_terminal_turns=TERMINAL_SPIN_TURNS,
                                                     distance_m=distance, burst_decisions=TERMINAL_BURST_DECISIONS)
        self._terminal_burst -= 1
        for row in result['candidates']:
            row['alignment_dropped_terminal'] = True
            row['utility_m'] = row['position_contact_utility_m']
        selected = max(range(len(result['candidates'])), key=lambda i: result['candidates'][i]['utility_m'])
        action = result['candidates'][selected]['action']
        result.update(action=action, action_index=selected, requested_command=candidate_commands(action)[0],
                      selection_objective='terminal_position_progress_minus_contact', terminal_alignment_dropped=True,
                      terminal_radius_m=TERMINAL_RADIUS_M)
        return result

    def __init__(self, *args, **kwargs):
        self._terminal_spin = (None, 0)
        self._terminal_turns, self._terminal_burst = 0, 0
        super().__init__(*args, **kwargs)

    def _select_clear_prediction(self, selected, prediction, snapshot, position, rotation):
        result = super()._select_clear_prediction(selected, prediction, snapshot, position, rotation)
        if result.get('rule') != 'terminal_measured_heading_then_existing_forward_pulse' or result['action'] not in ('left_turn', 'right_turn'):
            self._terminal_spin = (None, 0)
            return result
        direction, count = self._terminal_spin
        count = count+1 if direction == result['action'] else 1
        self._terminal_spin = (result['action'], count)
        forward = next(r for r in result['candidates'] if r['action'] == 'forward')
        if count < TERMINAL_SPIN_TURNS or not forward['eligible']:
            return result
        result.update(action='forward', action_index=ACTIONS.index('forward'), requested_command=candidate_commands('forward')[0],
                      command_duration_ns=TERMINAL_PULSE_NS)
        result['terminal_translation_pulse'] = dict(result['terminal_translation_pulse'], selected_translation_pulse=True)
        result['dev_terminal_spin_break'] = dict(consecutive_same_direction_turns=count, turn_direction=direction,
                                                 waypoint_body_xy_m=result['waypoint_body_xy_m'])
        self._terminal_spin = (None, 0)
        return result


TERMINAL_SPIN_TURNS, TERMINAL_PULSE_NS, TERMINAL_BURST_DECISIONS = 15, 100_000_000, 3
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
REACTIVE_ESCAPE_FLOOR_M, REACTIVE_ESCAPE_GAIN_M, REACTIVE_ESCAPE_SECONDS = .40, .005, .4


def plain(value):
    """JSON-safe copy (the installed strict JSON writer rejects repr fallbacks)."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    return repr(value)


def requested_endpoint(p, yaw, command, seconds, steps=20):
    """Unicycle integration of one constant requested command, map frame."""
    x, y, heading = float(p[0]), float(p[1]), yaw
    dt = seconds/steps
    for _ in range(steps):
        x += (command[0]*math.cos(heading)-command[1]*math.sin(heading))*dt
        y += (command[0]*math.sin(heading)+command[1]*math.cos(heading))*dt
        heading += command[2]*dt
    return np.array([x, y])


def observed_distance(point, cells):
    distances = segment_cell_distances(point, point, cells)
    return float(distances.min()) if len(distances) else math.inf


class DeadlockEscapeMixin:
    def __init__(self, *args, **kwargs):
        self._deadlock_holds, self._escape_armed = 0, None
        super().__init__(*args, **kwargs)

    def _select_clear_prediction(self, selected, prediction, snapshot, position, rotation):
        result = super()._select_clear_prediction(selected, prediction, snapshot, position, rotation)
        if 'memory_forecast_candidates' not in result and 'current_nominal_disk_clear' in result:
            return self._reactive_escape(result, snapshot, position, rotation)
        rows = result.get('memory_forecast_candidates')
        if not rows or result['action'] != 'hold':
            self._deadlock_holds, self._escape_armed = 0, None
            return result
        utilities = {r['action']: r['utility_m'] for r in result.get('scan_utilities', result['candidates'])}
        movable = [r for r in rows if r['action'] != 'hold' and r['action'] in utilities and r['nominal_predicted_path_clear']]
        if movable or (result.get('clearance_turn') or {}).get('active'):
            self._deadlock_holds, self._escape_armed = 0, None
            return result
        self._deadlock_holds += 1
        if self._deadlock_holds < ESCAPE_AFTER:
            return result
        current = next(r['minimum_predicted_path_clearance_m'] for r in rows if r['action'] == 'hold')
        floor = max(ESCAPE_FLOOR_M, (current or ESCAPE_FLOOR_M)-ESCAPE_SLACK_M)
        options = [i for i, r in enumerate(rows) if r['action'] in ('left_turn', 'right_turn') and r['action'] in utilities
                   and r['minimum_predicted_path_clearance_m'] is not None and r['minimum_predicted_path_clearance_m'] >= floor]
        if not options:
            if hasattr(self, '_request_backup'):
                self._request_backup('deadlock_no_clear_turn')
            return result
        goal = result.get('waypoint_body_xy_m')
        side = 'left_turn' if goal is None or math.atan2(goal[1], goal[0]) > 0 else 'right_turn'
        index = max(options, key=lambda i: (rows[i]['action'] == side, rows[i]['minimum_predicted_path_clearance_m']))
        choose(result, index)
        self._escape_armed = 'turn'
        result['dev_deadlock_escape'] = dict(consecutive_no_eligible_holds=self._deadlock_holds, current_clearance_m=current,
                                             clearance_floor_m=floor, action=rows[index]['action'], dispatch_disk_radius_m=ESCAPE_FLOOR_M)
        return result

    def _reactive_escape(self, result, snapshot, position, rotation):
        """C2: move away from stored obstacles when the 0.45-m current-clearance gate holds forever."""
        if result['action'] != 'hold' or result['current_nominal_disk_clear'] is not False:
            self._deadlock_holds, self._escape_armed = 0, None
            return result
        self._deadlock_holds += 1
        current = result.get('current_stored_clearance_m')
        if self._deadlock_holds < ESCAPE_AFTER or current is None or current < REACTIVE_ESCAPE_FLOOR_M:
            return result
        clearance = cached_clearance(snapshot.fine_occupied)
        p = np.asarray(position, float)[:2]
        yaw = math.atan2(rotation[1][0], rotation[0][0])
        desired = {r['action']: r['normalized_command_distance_squared'] for r in result['candidates']}
        moves = []
        for action in ('forward', 'left_arc', 'right_arc'):
            end = requested_endpoint(p, yaw, candidate_commands(action)[0], REACTIVE_ESCAPE_SECONDS)
            path, final = clearance.minimum(p, end), clearance.minimum(end, end)
            if path is not None and path >= current-ESCAPE_SLACK_M and final >= current+REACTIVE_ESCAPE_GAIN_M:
                moves.append((desired[action], action, path, final))
        if moves:
            _, action, path, final = min(moves)
            detail = dict(kind='translation', path_clearance_m=path, end_clearance_m=final)
        else:
            headings = np.linspace(-math.pi, math.pi, 16, endpoint=False)
            probes = [p+.1*np.array([math.cos(yaw+h), math.sin(yaw+h)]) for h in headings]
            gains = [clearance.minimum(q, q) for q in probes]
            best = int(np.argmax(gains))
            if gains[best] < current+REACTIVE_ESCAPE_GAIN_M:
                if hasattr(self, '_request_backup'):
                    self._request_backup('reactive_deadlock_no_move')
                return result
            action = 'left_turn' if headings[best] > 0 else 'right_turn'
            detail = dict(kind='turn_toward_clearance', relative_heading_rad=float(headings[best]), clearance_there_m=float(gains[best]))
        index = ACTIONS.index(action)
        result.update(action=action, action_index=index, requested_command=candidate_commands(action)[0],
                      command_duration_ns=int(REACTIVE_ESCAPE_SECONDS*1e9))
        self._escape_armed = 'reactive'
        result['dev_deadlock_escape'] = dict(consecutive_no_eligible_holds=self._deadlock_holds, current_clearance_m=current,
                                             action=action, dispatch_disk_radius_m=REACTIVE_ESCAPE_FLOOR_M, **detail)
        return result

    def request(self, *, now_ns):
        result = super().request(now_ns=now_ns)
        if not self._escape_armed or result['reason'] not in ('CURRENT_OBSERVED_OBSTACLE_VETO', 'COMMAND_WINDOW_VETO_LATCHED'):
            return result
        with self.lock:
            live = [q for q in self.plans if q.dispatch_ns <= now_ns < q.expires_ns]
            plan, current = (live[-1] if live else None), self.latest_obstacles
        if plan is None or current is None or not any(plan.command):
            return result
        translating = any(plan.command[:2])
        if translating and self._escape_armed != 'reactive':
            return result
        radius = REACTIVE_ESCAPE_FLOOR_M if self._escape_armed == 'reactive' else ESCAPE_FLOOR_M
        p, R = np.asarray(current.position_map), np.asarray(current.rotation_map_from_body)
        cells = sorted(current.occupied)
        endpoint = p
        if translating:
            seconds = (plan.expires_ns-now_ns)/1e9
            endpoint = p+R@np.array([plan.command[0]*seconds, plan.command[1]*seconds, 0.])
        check = nominal_connector(p[:2], endpoint[:2], cells, radius_m=radius)
        away = not translating or observed_distance(endpoint[:2], cells) >= observed_distance(p[:2], cells)
        if not check['nominal_disk_connector_clear'] or not away:
            return result
        with self.lock:
            self.rejected_windows.pop(plan.observed_ns, None)
        return result | dict(requested_command=plan.request(now_ns=now_ns, fresh_observation_allows_motion=True),
                             reason='CURRENT_NOMINAL_OBSTACLE_TEST_PASSED', dev_escape_dispatch=dict(
                                 kind=self._escape_armed, frozen_reason=result['reason'], disk_radius_m=radius,
                                 moves_away_from_observed=away, connector=check))


STALL_NS, STALL_RADIUS_M = 30_000_000_000, .15
STALL_EXCLUSION_NS, STALL_EXCLUSION_RADIUS_M = 90_000_000_000, 1.0


class StickyExclusions(set):
    """The visit logic clears its exclusions on new map evidence; watchdog exclusions persist until expiry."""

    def __init__(self, items=()):
        super().__init__(items)
        self.sticky = {}

    def clear(self):
        super().clear()
        self.update(self.sticky)

    def expire(self, now_ns):
        for cell in [c for c, until in self.sticky.items() if until <= now_ns]:
            del self.sticky[cell]
            self.discard(cell)


class StallWatchdogMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stall_window = None
        self.dev_stall_events = []
        self._backup_sites = []

    def _route(self, snapshot, evidence, goal, *, measured_ns):
        from lewm.observed_floor_waypoint_development import centre
        route = super()._route(snapshot, evidence, goal, measured_ns=measured_ns)
        visits = getattr(self, 'frontier_visits', None)
        if visits is None:
            return route
        if not isinstance(visits.excluded, StickyExclusions):
            visits.excluded = StickyExclusions(visits.excluded)
        visits.excluded.expire(measured_ns)
        outbound = self.mission_latest is None or self.mission_latest['phase'] == 'OUTBOUND'
        if not outbound or route.get('status') != 'OBSERVED_FLOOR_ROUTE_TO_FRONTIER':
            self._stall_window = None
            return route
        p, _, _ = self._pose(evidence, identity=(0, 0, 0), now_ns=measured_ns)
        q = (np.asarray(snapshot.map_from_initial)@p)[:2]
        window = self._stall_window
        if window is None or np.linalg.norm(q-window['start_xy']) > STALL_RADIUS_M:
            self._stall_window = dict(start_ns=measured_ns, start_xy=q)
            return route
        if measured_ns-window['start_ns'] < STALL_NS:
            return route
        if hasattr(self, '_request_backup') and not any(np.linalg.norm(q-np.asarray(b)) <= BACKUP_REPEAT_RADIUS_M for b in self._backup_sites):
            self._backup_sites.append(q.tolist())
            self._request_backup('stall')
            event = dict(measured_ns=measured_ns, stalled_since_ns=window['start_ns'], position_map_xy_m=q.tolist(), remedy='backup')
            self.dev_stall_events.append(event)
            print(json.dumps(plain(dict(dev_stall=event))), flush=True)
            self._stall_window = dict(start_ns=measured_ns, start_xy=q)
            return route
        target = tuple(route['route_cells'][-1])
        abandoned = None
        if visits.visit is not None:
            v = visits.visit
            abandoned = dict(target_cell=v.get('target_cell'), unknown_neighbour=v.get('unknown_neighbour'))
            viewpoint = v.get('camera_viewpoint', {}).get('viewpoint_map_xy_m')
            if viewpoint is not None and v.get('unknown_neighbour') is not None:
                tried = visits.attempted.setdefault(tuple(v['unknown_neighbour']), set())
                tried.update(c for c in snapshot.floor if np.linalg.norm(centre(c)-np.asarray(viewpoint)) <= .10)
            if v.get('target_cell') is not None:
                target = tuple(v['target_cell'])
            visits._finish(snapshot, measured_ns, 'DEV_NO_PROGRESS_STALL', False)
        cells = {c for c in snapshot.floor if np.linalg.norm(centre(c)-centre(target)) <= STALL_EXCLUSION_RADIUS_M} | {target}
        for cell in cells:
            visits.excluded.sticky[cell] = measured_ns+STALL_EXCLUSION_NS
        visits.excluded.update(cells)
        event = dict(measured_ns=measured_ns, stalled_since_ns=window['start_ns'], position_map_xy_m=q.tolist(),
                     frontier_cell=list(target), excluded_cells=len(cells), abandoned_visit=abandoned,
                     exclusion_until_ns=measured_ns+STALL_EXCLUSION_NS, remedy='frontier_exclusion')
        self._stall_window = dict(start_ns=measured_ns, start_xy=q)
        rerouted = super()._route(snapshot, evidence, goal, measured_ns=measured_ns)
        if rerouted.get('status') not in ('OBSERVED_FLOOR_ROUTE_TO_FRONTIER', 'OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL'):
            # Never retire the last reachable frontier (C1 validation 13: the exclusion left no
            # frontier, and the robot spun for the 90-s exclusion instead of exploring).
            for cell in cells:
                visits.excluded.sticky.pop(cell, None)
                visits.excluded.discard(cell)
            event.update(remedy='exclusion_undone_no_other_route', status_after_exclusion=rerouted.get('status'))
            rerouted = super()._route(snapshot, evidence, goal, measured_ns=measured_ns)
        self.dev_stall_events.append(event)
        print(json.dumps(plain(dict(dev_stall=event))), flush=True)
        rerouted['dev_stall_reroute'] = plain(event)
        return rerouted


BACKUP_SPEED_MPS, BACKUP_DECISIONS, BACKUP_STEP_S = .10, 5, .4
BACKUP_FLOOR_M, BACKUP_REPEAT_RADIUS_M, BACKUP_OBSERVED_SLACK_M = .40, .30, .005


class ScriptedBackupMixin:
    def __init__(self, *args, **kwargs):
        self._backup, self._backup_request, self._backup_store, self._backup_plans = None, None, False, set()
        super().__init__(*args, **kwargs)

    def _request_backup(self, reason):
        if self._backup is None and self._backup_request is None:
            self._backup_request = reason

    def _select_clear_prediction(self, selected, prediction, snapshot, position, rotation):
        result = super()._select_clear_prediction(selected, prediction, snapshot, position, rotation)
        self._backup_store = False
        if self._backup is None and self._backup_request is None:
            return result
        if self._backup is None:
            self._backup = dict(reason=self._backup_request, step=0)
            self._backup_request = None
        backup = self._backup
        clearance = cached_clearance(snapshot.fine_occupied)
        p = np.asarray(position, float)[:2]
        yaw = math.atan2(rotation[1][0], rotation[0][0])
        end = p-BACKUP_SPEED_MPS*BACKUP_STEP_S*np.array([math.cos(yaw), math.sin(yaw)])
        current, path = clearance.minimum(p, p), clearance.minimum(p, end)
        on_floor = all(tuple(c) in snapshot.floor for c in segment_cells(p, end))
        clear = path is None or (path >= BACKUP_FLOOR_M and (current is None or path >= current-.01))
        detail = dict(reason=backup['reason'], step=backup['step']+1, of=BACKUP_DECISIONS, current_clearance_m=current,
                      path_clearance_m=path, on_observed_floor=on_floor)
        if not (on_floor and clear):
            result['dev_backup_aborted'] = detail
            self._backup = None
            return result
        if 'dev_deadlock_escape' in result:
            result['dev_deadlock_escape_overridden_by_backup'] = result.pop('dev_deadlock_escape')
        index = ACTIONS.index('hold')
        result.update(action='hold', action_index=index, requested_command=[-BACKUP_SPEED_MPS, 0., 0.],
                      command_duration_ns=int(BACKUP_STEP_S*1e9))
        result['dev_backup'] = detail
        self._backup_store = True
        backup['step'] += 1
        if backup['step'] >= BACKUP_DECISIONS:
            self._backup = None
        return result

    def _store_plan(self, plan, completed, prefix):
        if self._backup_store:
            plan = replace(plan, command=(-BACKUP_SPEED_MPS, 0., 0.))
            self._backup_plans.add(plan.observed_ns)
            self._backup_store = False
        return super()._store_plan(plan, completed, prefix)

    def request(self, *, now_ns):
        result = super().request(now_ns=now_ns)
        if result['reason'] not in ('CURRENT_OBSERVED_OBSTACLE_VETO', 'COMMAND_WINDOW_VETO_LATCHED'):
            return result
        with self.lock:
            live = [q for q in self.plans if q.dispatch_ns <= now_ns < q.expires_ns]
            plan, current = (live[-1] if live else None), self.latest_obstacles
        if plan is None or current is None or plan.observed_ns not in self._backup_plans:
            return result
        p, R = np.asarray(current.position_map), np.asarray(current.rotation_map_from_body)
        cells = sorted(current.occupied)
        seconds = (plan.expires_ns-now_ns)/1e9
        endpoint = p+R@np.array([plan.command[0]*seconds, 0., 0.])
        check = nominal_connector(p[:2], endpoint[:2], cells, radius_m=BACKUP_FLOOR_M)
        away = observed_distance(endpoint[:2], cells) >= observed_distance(p[:2], cells)-BACKUP_OBSERVED_SLACK_M
        if not check['nominal_disk_connector_clear'] or not away:
            return result
        with self.lock:
            self.rejected_windows.pop(plan.observed_ns, None)
        return result | dict(requested_command=plan.request(now_ns=now_ns, fresh_observation_allows_motion=True),
                             reason='CURRENT_NOMINAL_OBSTACLE_TEST_PASSED',
                             dev_backup_dispatch=dict(frozen_reason=result['reason'], disk_radius_m=BACKUP_FLOOR_M, connector=check))


class PoseLossRecordMixin:
    """MeasuredLatencyRuntime._track, plus a worker.log record of the tracker failure chain."""

    def _track(self, packet):
        raw = self.pose_executor.submit(pose_update, replace(packet, history=())).result()
        if raw.get('current_pose') is None or raw.get('failure') is not None:
            try:
                print(json.dumps(plain(dict(dev_pose_loss=dict(frame=packet.frame, measured_ns=packet.measured_ns, status=raw.get('status'),
                                                               failure=raw.get('failure'), contact=raw.get('contact'))))), flush=True)
            except Exception as error:  # the record must never replace the pose-loss fault
                print(f'dev_pose_loss frame={packet.frame} record failed: {error!r}', flush=True)
            raise ValueError('measured visual pose unavailable')
        self.clock_ns()
        self.queues['registration'].put_nowait((packet, raw))


FIXES = {'terminal': TerminalPositionScoringMixin, 'latch': LatchTimeoutMixin, 'deadlock': DeadlockEscapeMixin,
         'pose': PoseLossRecordMixin, 'stall': StallWatchdogMixin, 'backup': ScriptedBackupMixin}


_COVERAGE_FIX = contextvars.ContextVar('dev_coverage_fix', default=False)
_FROZEN_FILTER = getattr(coverage_rule.filter_translation, 'frozen', coverage_rule.filter_translation)


class ObservedFloorView:
    """The snapshot with observed occupied cells counted as observed by the coverage rule."""

    def __init__(self, snapshot):
        self._snapshot = snapshot
        self.floor = snapshot.floor | snapshot.occupied

    def __getattr__(self, name):
        return getattr(self._snapshot, name)


def _filter_translation(selection, prediction, snapshot, position, rotation):
    if _COVERAGE_FIX.get():
        snapshot = ObservedFloorView(snapshot)
    return _FROZEN_FILTER(selection, prediction, snapshot, position, rotation)


_filter_translation.frozen = _FROZEN_FILTER
coverage_rule.filter_translation = _filter_translation  # behaviour changes only under CoverageObservedObstacleMixin


class CoverageObservedObstacleMixin:
    def _select_clear_prediction(self, selected, prediction, snapshot, position, rotation):
        token = _COVERAGE_FIX.set(True)
        try:
            return super()._select_clear_prediction(selected, prediction, snapshot, position, rotation)
        finally:
            _COVERAGE_FIX.reset(token)


FIXES['coverage'] = CoverageObservedObstacleMixin


NOISE_PROFILE = np.arange(1, 9)/7.  # forecast error grows linearly with horizon; 1.0 at the scored 700 ms


def degradation_mixin(spec):
    """Degrade the forecast the planner scores (Andrew, 2 October: forecast-sensitivity experiment).

    'scale:S' multiplies every candidate's predicted displacement and heading change by S.
    Structured errors (Andrew, 2 October; copying C3's closed-loop failure modes):
    'fwdscale:S' multiplies the predicted displacement of the translating candidates only
    (forward, left arc, right arc; heading unchanged): S < 1 under-predicts forward travel.
    'turnscale:S' multiplies the predicted displacement and heading change of the in-place
    turn candidates only: S > 1 over-predicts turns. Other candidates are unchanged.
    'noise:E' adds, per decision and candidate, a 2-D Gaussian displacement error whose median
    magnitude at 700 ms is E mm, and a heading error whose median magnitude at 700 ms is E/10
    degrees (10 mm with 1 degree), both growing linearly with horizon. Noise is seeded by the
    level and the frame, so a mission is reproducible. The degraded forecast is logged in the
    decision's motion correction ('dev_degraded_forecast_xy_yaw') so it can be scored like
    C3's and C4's.
    """
    kind, value = spec.split(':')
    value = float(value)
    if kind not in ('scale', 'noise', 'fwdscale', 'turnscale'):
        raise ValueError('degradation must be scale:S, noise:E_mm, fwdscale:S or turnscale:S')
    translating = [ACTIONS.index(a) for a in ('forward', 'left_arc', 'right_arc')]
    turning = [ACTIONS.index(a) for a in ('left_turn', 'right_turn')]

    class ForecastDegradationMixin:
        degradation = dict(kind=kind, value=value)

        def _correct_prediction(self, prediction, packet, evidence, prefix):
            prediction, correction = super()._correct_prediction(prediction, packet, evidence, prefix)
            p = np.array(prediction, float, copy=True)
            yaw = np.arctan2(p[..., 2], p[..., 3])
            if kind == 'scale':
                p[..., :2] *= value
                yaw = yaw*value
            elif kind == 'fwdscale':
                p[translating, :, :2] *= value
            elif kind == 'turnscale':
                p[turning, :, :2] *= value
                yaw[turning] = yaw[turning]*value
            else:
                rng = np.random.default_rng([20261002, int(round(value*1000)), int(packet.frame)])
                sigma_xy = value/1000/math.sqrt(2*math.log(2))     # median |N(0, s^2 I2)| = s*sqrt(2 ln 2)
                sigma_yaw = math.radians(value/10)/0.6744897501960817  # median |N(0, s^2)| = 0.6745 s
                p[..., :2] += rng.normal(0, sigma_xy, size=(p.shape[0], 1, 2))*NOISE_PROFILE[None, :, None]
                yaw = yaw+rng.normal(0, sigma_yaw, size=(p.shape[0], 1))*NOISE_PROFILE[None, :]
            p[..., 2], p[..., 3] = np.sin(yaw), np.cos(yaw)
            correction = dict(correction or {})
            correction['dev_degraded_forecast_xy_yaw'] = np.stack((p[..., 0], p[..., 1], yaw), axis=-1).tolist()
            correction['dev_degradation'] = dict(kind=kind, value=value)
            return p, correction

    ForecastDegradationMixin.__name__ = f'ForecastDegradation_{kind}_{value:g}'.replace('.', 'p')
    return ForecastDegradationMixin


# Calibrated clearance margin (Andrew, 2 October: calibrated-margin experiment). The forecast-
# based action check (memory forecast clearance, whose path distances the turn reserve and the
# reserve-recovery modes reuse) and the route-target lookahead read remembered-cell distances
# through their own module's `cached_clearance`. Inside `margin_mixin` those distances are
# reduced by the controller's calibrated bound, so every requirement r becomes r + margin; the
# logged distances are the reduced ones, and the selection records the margin. The stopping
# projection, the dispatch depth stop, the coverage rule and the routing graph keep their own
# bindings and are unchanged. Outside the mixin the behaviour is the frozen one.
_CLEARANCE_MARGIN = contextvars.ContextVar('dev_clearance_margin_m', default=0.)


class _MarginClearance:
    def __init__(self, inner, margin):
        self._inner, self._margin = inner, margin

    def minimum(self, start, end):
        value = self._inner.minimum(start, end)
        return None if value is None else value-self._margin

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _margin_cached_clearance(cells):
    inner = cached_clearance(cells)
    margin = _CLEARANCE_MARGIN.get()
    return _MarginClearance(inner, margin) if margin else inner


from lewm import clearance_lookahead_development as _lookahead  # noqa: E402
from lewm import memory_forecast_clearance_development as _memory_check  # noqa: E402
MARGIN_MODULES = (_memory_check, _lookahead)
for _module in MARGIN_MODULES:
    _module.cached_clearance = _margin_cached_clearance  # behaviour changes only under margin_mixin


def margin_mixin(margin_m, label):
    """Inflate the forecast-based clearance requirements by a calibrated margin (metres)."""
    if not np.isfinite(margin_m) or not 0. < margin_m <= .10:
        raise ValueError('calibrated margin must be in (0, 0.10] m')

    class ClearanceMarginMixin:
        clearance_margin = dict(margin_m=margin_m, label=label)

        def _select_clear_prediction(self, selected, prediction, snapshot, position, rotation):
            token = _CLEARANCE_MARGIN.set(margin_m)
            try:
                result = super()._select_clear_prediction(selected, prediction, snapshot, position, rotation)
            finally:
                _CLEARANCE_MARGIN.reset(token)
            selection = result[0] if isinstance(result, tuple) else result
            selection['dev_clearance_margin'] = dict(margin_m=margin_m, label=label, logged_distances_reduced_by_margin=True)
            return result

        def _route_target(self, *args, **kwargs):
            token = _CLEARANCE_MARGIN.set(margin_m)
            try:
                return super()._route_target(*args, **kwargs)
            finally:
                _CLEARANCE_MARGIN.reset(token)

    ClearanceMarginMixin.__name__ = f'ClearanceMargin_{label}'
    return ClearanceMarginMixin


# Andrew (30 Sep evening): the preliminary run drives every controller twice, recovery on (the
# full development system) and recovery off (the controller's own choices on the frozen V4
# harness). Every behavioural fix overrides a controller choice, so all of them are recovery;
# `pose` only records the tracker failure chain and stays on in both.
RECOVERY_FIXES = ('backup', 'deadlock', 'latch', 'stall', 'terminal')
DIAGNOSTIC_FIXES = ('pose',)
# Andrew (2 Oct): recovery off is the default from now on; the coverage-rule fix applies in both settings.
HARNESS_FIXES = ('coverage',)
DEFAULT_RECOVERY = 'off'


def fixes_for(recovery, harness=True):
    """The fix list for --recovery on|off. harness=False gives the preliminary run's sets (no coverage fix)."""
    if recovery not in ('on', 'off'):
        raise ValueError('recovery must be on or off')
    return sorted(DIAGNOSTIC_FIXES+(HARNESS_FIXES if harness else ())+(RECOVERY_FIXES if recovery == 'on' else ()))


def compose(fixes, base, extra=()):
    """The frozen startup mixin with the requested fixes (and any extra mixins, outermost) ahead of it."""
    unknown = set(fixes)-set(FIXES)
    if unknown:
        raise ValueError(f'unknown fixes: {sorted(unknown)}')
    bases = tuple(extra)+tuple(FIXES[f] for f in sorted(fixes))+(base,)
    return type('DevStartupRecoveryRuntimeMixin', bases, {})


def check_track_override(runtime):
    """`pose` replaces MeasuredLatencyRuntime._track; refuse if anything else sits between them."""
    owners = [k for k in runtime.__mro__ if '_track' in k.__dict__]
    if owners and owners[0] is PoseLossRecordMixin and owners[1].__name__ != 'MeasuredLatencyRuntime':
        raise TypeError(f'pose fix would shadow {owners[1].__module__}.{owners[1].__name__}._track')
