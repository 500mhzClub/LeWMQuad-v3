"""Distinct measured release/settle alignment; no predecessor result changes."""
import math

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.continuation_branch_development import wrap
from lewm.measured_line_integral_navigation_development import (
    BoundedIntegralAlignment, MeasuredLineIntegralNavigation)


class ReleaseAwareAlignment(BoundedIntegralAlignment):
    """Drive to an inner target, then verify actual zero-command settling.

    The outer tolerance/rate/dwell and total deadline remain unchanged. The
    extra half-second release observation is stricter, not a success relaxation.
    No constant release-displacement correction or true pose is used.
    """
    def __init__(self, direction_initial_body):
        super().__init__(direction_initial_body)
        self.phase = 'APPROACH'
        self.release_ns = None
        self.failed = False
        self.release_attempts = 0

    def observe(self, packet, attitude, *, now_ns):
        if self.failed: raise SensorContractError('release alignment fault latched')
        try:
            return self._observe_release(packet, attitude, now_ns=now_ns)
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            self.failed = True
            raise SensorContractError('invalid release alignment observation; apply zero') from error

    def _observe_release(self, packet, attitude, *, now_ns):
        if (self.terminal or type(now_ns) is not int or now_ns < 0
                or (self.last_ns is not None and now_ns-self.last_ns != 100_000_000)):
            raise SensorContractError('active consecutive alignment required')
        if attitude['decision_ns'] != now_ns or packet['sensor_state']['decision_ns'] != now_ns:
            raise SensorContractError('current causal attitude required')
        rotation = np.asarray(attitude['rotation_initial_body_from_current_body'], dtype=float)
        gyro = packet['sensor_state']['sensed']['gyro']
        values, valid = np.asarray(gyro['values']), np.asarray(gyro['valid'])
        if (rotation.shape != (3,3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-7, rtol=0)
                or abs(np.linalg.det(rotation)-1.) > 1e-7
                or values.ndim != 2 or values.shape[1:] != (3,) or len(values) == 0
                or valid.shape != values.shape or valid.dtype.kind != 'b'
                or not valid[-1].all() or not np.isfinite(values[-1]).all()):
            raise SensorContractError('proper observed attitude and valid gyro required')
        forward = rotation[:,0]
        if np.linalg.norm(forward[:2]) < .2: raise SensorContractError('near-vertical heading')
        error = wrap(math.atan2(self.direction[1],self.direction[0])-math.atan2(forward[1],forward[0]))
        derivative = np.cross(rotation@values[-1], forward)
        rate = float((forward[0]*derivative[1]-forward[1]*derivative[0])/(forward[0]**2+forward[1]**2))
        if self.start_ns is None: self.start_ns = now_ns
        quiet = abs(error) <= .02 and abs(rate) <= .1
        status, yaw = 'ALIGNING', 0.
        if self.phase == 'APPROACH' and abs(error) <= .005 and abs(rate) <= .1:
            self.phase = 'SETTLE'; self.release_ns = now_ns
            self.release_attempts += 1; self.quiet_since = now_ns
            self.integral_command = 0.
        if self.phase == 'SETTLE':
            if quiet:
                if self.quiet_since is None: self.quiet_since = now_ns
            else: self.quiet_since = None
            if now_ns-self.release_ns >= 500_000_000:
                if self.quiet_since is not None and now_ns-self.quiet_since >= 300_000_000:
                    status = 'COMPLETE'
                elif not quiet:
                    # Observe the release transient before correcting again;
                    # keep the original deadline, not a fresh attempt budget.
                    self.phase = 'APPROACH'; self.quiet_since = None
                    self.previous_error = None
        if self.phase == 'APPROACH':
            if self.previous_error is not None and error*self.previous_error < 0:
                self.integral_command = 0.
            raw = 1.5*error+self.integral_command
            if abs(raw) < .35 or raw*error < 0:
                self.integral_command = float(np.clip(self.integral_command+.4*error*.1,-.12,.12))
            yaw = float(np.clip(1.5*error+self.integral_command,-.35,.35))
        if status != 'COMPLETE' and now_ns-self.start_ns >= 12_000_000_000:
            status = 'FAILED_TIMEOUT'
        self.terminal = status != 'ALIGNING'
        if self.terminal: yaw = 0.; self.integral_command = 0.
        self.last_ns = now_ns; self.previous_error = error
        return {'status':status, 'decision_ns':now_ns, 'heading_error_rad':error,
            'projected_heading_rate_rad_s':rate, 'requested_command':[0.,0.,yaw],
            'translation_compensated':False, 'clearance_qualified':False,
            'integral_command_rad_s':self.integral_command, 'phase':self.phase,
            'release_started_ns':self.release_ns, 'release_attempts':self.release_attempts,
            'quiet_since_ns':self.quiet_since, 'inner_target_rad':.005,
            'controller':'release_aware_alignment_development_v1'}


class ReleaseAwareNavigation(MeasuredLineIntegralNavigation):
    def __init__(self, method, geometry, template=None, *, memory_arm):
        if method != 'release_aware' or template is not None:
            raise ValueError('distinct release-aware development method required')
        super().__init__('measured_line_integral',geometry,memory_arm=memory_arm)

    def observe_rgbd(self, packet, fast_packet, depth, relative, *, now_ns):
        try:
            decision = super().observe_rgbd(packet,fast_packet,depth,relative,now_ns=now_ns)
            if self.stage == 'ALIGN' and not isinstance(self.alignment,ReleaseAwareAlignment):
                if self.alignment is None or self.alignment.start_ns is not None:
                    raise SensorContractError('only replace an unstarted alignment')
                self.alignment = ReleaseAwareAlignment(self.selected['direction_initial_body'])
            decision['local_controller'] = 'release_aware_navigation_development_v1'
            return decision
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError) as error:
            self._fail('FAILED_SENSOR',self.last_ns if self.last_ns is not None else 0)
            raise SensorContractError('release-aware navigation failure; apply zero') from error
