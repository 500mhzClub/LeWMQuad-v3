"""Observed dwell requires low measured 3D motion as well as zero requests.

Finite differences of admitted visual positions are interval-average motion,
not bounds on instantaneous speed. Independent continuous-speed verification
remains required. This successor preserves the original mission and its budget.
"""
from copy import deepcopy
import numpy as np
from lewm.observed_round_trip_mission_development import ObservedRoundTripMission

MAXIMUM_OBSERVED_INTERVAL_SPEED_M_S = .05


class MeasuredSettlingRoundTripMission(ObservedRoundTripMission):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous_visual_position = None

    def advance(self, position_initial_body_m, *, frame, now_ns, previous_requested_command):
        if self.terminal is not None:
            return deepcopy(self.last)
        try:
            p = np.asarray(position_initial_body_m, float)
            if (p.shape != (3,) or not np.isfinite(p).all() or type(frame) is not int
                    or frame != self.frame+1 or type(now_ns) is not int
                    or now_ns != 1_500_000_000+frame*100_000_000):
                raise ValueError('consecutive finite admitted 3D visual positions required for settling')
            previous = self.previous_visual_position
            speed = None if previous is None else float(np.linalg.norm(p-previous)/.1)
            motion_quiet = speed is not None and speed <= MAXIMUM_OBSERVED_INTERVAL_SPEED_M_S
            if not motion_quiet:
                # Prevent the original dwell rule from counting this interval.
                # Keep actual coordinates and the actual command unchanged.
                self.quiet = 0
                self.was_within = False
            result = super().advance(p[:2], frame=frame, now_ns=now_ns,
                previous_requested_command=previous_requested_command)
            if result['failure'] is None:
                self.previous_visual_position = p.copy()
            receipt = dict(previous_frame=None if previous is None else frame-1,
                current_frame=frame, measured_ns=now_ns,
                previous_position_initial_body_m=None if previous is None else previous.tolist(),
                current_position_initial_body_m=p.tolist(),
                observed_interval_speed_m_s=speed,
                maximum_observed_interval_speed_m_s=MAXIMUM_OBSERVED_INTERVAL_SPEED_M_S,
                measured_motion_quiet=motion_quiet,
                motion_source='consecutive_admitted_floor_registered_visual_positions',
                continuous_speed_bound=False, pose_uncertainty_calibrated=False,
                native_state_used=False, command_integrated_position=False)
            self.last = result | dict(measured_settling_required=True, observed_settling=receipt)
        except (ValueError, TypeError, KeyError) as error:
            self.terminal = 'SENSOR_OR_MISSION_FAILURE'; self.failure = str(error)
            self.last = dict(frame=self.frame, measured_ns=self.now_ns, phase=self.phase,
                active_goal_initial_body_xy_m=self.target().tolist(), terminal=self.terminal,
                failure=self.failure, hold_required=True, arrivals=deepcopy(self.arrivals),
                measured_settling_required=True, observed_settling=None,
                native_state_used=False, verified_round_trip=False)
        return deepcopy(self.last)
