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
  is abandoned (its viewpoint recorded as tried) and the frontier cell and its neighbours
  within STALL_EXCLUSION_RADIUS_M are excluded from target selection for
  STALL_EXCLUSION_NS. The exclusion survives the visit logic's clear-on-new-map rule.
  Exclusion only changes which frontier is selected; no cell is marked free or blocked.
- `pose`: record why visual pose was lost. The tracker is terminal after its first failure,
  and every downstream consumer (registration, map, routing) re-derives the pose from the
  tracker's own evidence chain, so re-anchoring means re-plumbing those validators. The one
  pose loss seen on this harness (C1 fresh-check 09) came at the end of a flipping latched
  turn, facing a featureless wall, so `latch` targets its cause. This fix prints the
  tracker's failure chain to worker.log and then faults exactly as before.
"""
from dataclasses import replace
import json
import math

import numpy as np

from lewm.clearance_turn_recovery_development import choose, wrap
from lewm.fine_stored_obstacle_routing_development import cached_clearance
from lewm.observed_geometry_refinement_development import nominal_connector, segment_cell_distances
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.mission_coordinate_metric_development import position_distance
from lewm.process_mapped_runtime_development import pose_update

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
STALL_EXCLUSION_NS, STALL_EXCLUSION_RADIUS_M = 90_000_000_000, .30


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
        target = tuple(route['route_cells'][-1])
        cells = {c for c in snapshot.floor if np.linalg.norm(centre(c)-centre(target)) <= STALL_EXCLUSION_RADIUS_M} | {target}
        abandoned = None
        if visits.visit is not None:
            v = visits.visit
            abandoned = dict(target_cell=v.get('target_cell'), unknown_neighbour=v.get('unknown_neighbour'))
            viewpoint = v.get('camera_viewpoint', {}).get('viewpoint_map_xy_m')
            if viewpoint is not None and v.get('unknown_neighbour') is not None:
                tried = visits.attempted.setdefault(tuple(v['unknown_neighbour']), set())
                tried.update(c for c in snapshot.floor if np.linalg.norm(centre(c)-np.asarray(viewpoint)) <= .10)
            cells |= {tuple(v['target_cell'])} if v.get('target_cell') is not None else set()
            visits._finish(snapshot, measured_ns, 'DEV_NO_PROGRESS_STALL', False)
        for cell in cells:
            visits.excluded.sticky[cell] = measured_ns+STALL_EXCLUSION_NS
        visits.excluded.update(cells)
        event = dict(measured_ns=measured_ns, stalled_since_ns=window['start_ns'], position_map_xy_m=q.tolist(),
                     frontier_cell=list(target), excluded_cells=len(cells), abandoned_visit=abandoned,
                     exclusion_until_ns=measured_ns+STALL_EXCLUSION_NS)
        self.dev_stall_events.append(event)
        print(json.dumps(plain(dict(dev_stall=event))), flush=True)
        self._stall_window = dict(start_ns=measured_ns, start_xy=q)
        rerouted = super()._route(snapshot, evidence, goal, measured_ns=measured_ns)
        rerouted['dev_stall_reroute'] = plain(event)
        return rerouted


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
         'pose': PoseLossRecordMixin, 'stall': StallWatchdogMixin}


def compose(fixes, base):
    """The frozen startup mixin with the requested fixes ahead of it."""
    unknown = set(fixes)-set(FIXES)
    if unknown:
        raise ValueError(f'unknown fixes: {sorted(unknown)}')
    bases = tuple(FIXES[f] for f in sorted(fixes))+(base,)
    return type('DevStartupRecoveryRuntimeMixin', bases, {})


def check_track_override(runtime):
    """`pose` replaces MeasuredLatencyRuntime._track; refuse if anything else sits between them."""
    owners = [k for k in runtime.__mro__ if '_track' in k.__dict__]
    if owners and owners[0] is PoseLossRecordMixin and owners[1].__name__ != 'MeasuredLatencyRuntime':
        raise TypeError(f'pose fix would shadow {owners[1].__module__}.{owners[1].__name__}._track')
