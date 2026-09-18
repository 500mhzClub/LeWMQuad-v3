"""Observed outbound/return state for future maze-controller integration.

Coordinates are instructions, never map or free-space observations. Arrivals
are candidates requiring independent native/hardware verification.
"""
from copy import deepcopy
import numpy as np
from lewm.matched_model_goal_probe_development import (
    WARMUP_TICKS, ARRIVAL_RADIUS_M, ARRIVAL_QUIET_INTERVALS)

MAX_NAVIGATION_TICKS = 4000


def point(value):
    p = np.asarray(value, float)
    if p.shape != (2,) or not np.isfinite(p).all() or np.max(np.abs(p)) > 4.9:
        raise ValueError('finite initial-frame mission point within map bounds required')
    return p.copy()


class ObservedRoundTripMission:
    def __init__(self, public_mission, *, navigation_ticks, arrival_radius_m=ARRIVAL_RADIUS_M):
        if (set(public_mission) != {'goal_initial_body_xy_m', 'return_initial_body_xy_m', 'require_return_after_goal'}
                or type(public_mission['require_return_after_goal']) is not bool
                or type(navigation_ticks) is not int or not 1 <= navigation_ticks <= MAX_NAVIGATION_TICKS):
            raise ValueError('explicit coordinate-only mission and bounded global budget required')
        if not np.isfinite(arrival_radius_m) or not 0 < arrival_radius_m <= ARRIVAL_RADIUS_M:
            raise ValueError('observed arrival radius must be positive and no greater than the physical target radius')
        self.arrival_radius_m=float(arrival_radius_m)
        self.outbound = point(public_mission['goal_initial_body_xy_m'])
        self.home = point(public_mission['return_initial_body_xy_m'])
        self.require_return = public_mission['require_return_after_goal']
        if self.require_return and np.linalg.norm(self.outbound-self.home) <= 2*ARRIVAL_RADIUS_M:
            raise ValueError('round-trip arrival regions must be distinct')
        self.navigation_ticks = navigation_ticks
        self.frame = -1; self.now_ns = None; self.phase = 'OUTBOUND'
        self.quiet = 0; self.was_within = False; self.terminal = None; self.failure = None
        self.arrivals = []; self.last = None

    def target(self):
        return (self.outbound if self.phase == 'OUTBOUND' else self.home).copy()

    def advance(self, position_initial_body_xy_m, *, frame, now_ns, previous_requested_command):
        if self.terminal is not None: return deepcopy(self.last)
        try:
            p = point(position_initial_body_xy_m); command = np.asarray(previous_requested_command, float)
            if (type(frame) is not int or frame != self.frame+1 or type(now_ns) is not int
                    or now_ns != 1_500_000_000+frame*100_000_000
                    or command.shape != (3,) or not np.isfinite(command).all()):
                raise ValueError('consecutive admitted pose clock and previous request required')
            evaluated = self.target(); phase = self.phase; distance = float(np.linalg.norm(p-evaluated))
            within = distance <= self.arrival_radius_m
            hold = frame < WARMUP_TICKS or within; transition = None; arrival = False
            if frame >= WARMUP_TICKS:
                self.quiet = self.quiet+1 if within and self.was_within and np.array_equal(command, np.zeros(3)) else 0
                if within and self.quiet >= ARRIVAL_QUIET_INTERVALS:
                    arrival = True
                    self.arrivals.append(dict(phase=phase, frame=frame, measured_ns=now_ns,
                        target_initial_body_xy_m=evaluated.tolist(), observed_position_initial_body_xy_m=p.tolist(),
                        observed_distance_m=distance, quiet_intervals=self.quiet, native_verified=False))
                    if phase == 'OUTBOUND' and self.require_return:
                        self.phase = 'RETURN'; self.quiet = 0; transition = 'OUTBOUND_TO_RETURN'
                    else:
                        self.terminal = 'OBSERVED_ROUND_TRIP_CANDIDATE' if self.require_return else 'OBSERVED_GOAL_CANDIDATE'
            if frame >= WARMUP_TICKS+self.navigation_ticks and self.terminal is None:
                self.terminal = 'MISSION_TICK_BUDGET_EXHAUSTED'; hold = True
            self.was_within = within if transition is None else False
            self.frame = frame; self.now_ns = now_ns
            self.last = dict(frame=frame, measured_ns=now_ns, phase=self.phase, phase_transition=transition,
                evaluated_phase=phase, evaluated_goal_initial_body_xy_m=evaluated.tolist(),
                active_goal_initial_body_xy_m=self.target().tolist(), observed_goal_distance_m=distance,
                observed_arrival_radius_m=self.arrival_radius_m,
                quiet_intervals=self.quiet, hold_required=hold or self.terminal is not None,
                arrival_confirmed_this_frame=arrival, arrivals=deepcopy(self.arrivals),
                terminal=self.terminal, failure=None, global_navigation_ticks=self.navigation_ticks,
                controller_state_reset_required=False, observed_map_reset_required=False,
                native_state_used=False, verified_round_trip=False)
        except (ValueError, TypeError, KeyError) as error:
            self.terminal = 'SENSOR_OR_MISSION_FAILURE'; self.failure = str(error)
            self.last = dict(frame=self.frame, measured_ns=self.now_ns, phase=self.phase,
                active_goal_initial_body_xy_m=self.target().tolist(), terminal=self.terminal,
                failure=self.failure, hold_required=True, arrivals=deepcopy(self.arrivals),
                native_state_used=False, verified_round_trip=False)
        return deepcopy(self.last)
