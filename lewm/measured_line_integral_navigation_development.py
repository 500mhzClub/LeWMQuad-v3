"""Unqualified successor: fixed-lookahead approach and bounded PI alignment.

No frozen experiment is changed. Acceptance, sensor observability, sampled
clearance and mission budgets remain those of the measured-region baseline.
These control changes require a new closed-loop experiment, not tape rescoring.
"""
import math

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.initially_aligned_continuation_development import FineInitialBearingAlignment
from lewm.measured_region_navigation_development import (
    MeasuredRegionNavigation, MeasuredRegionTraversal)


class BoundedIntegralAlignment(FineInitialBearingAlignment):
    """Same acceptance/deadline, with anti-windup and quiet-region reset."""
    def __init__(self, direction_initial_body):
        super().__init__(direction_initial_body)
        self.integral_command = 0.
        self.previous_error = None

    def observe(self, packet, attitude, *, now_ns):
        result = super().observe(packet, attitude, now_ns=now_ns)
        error = result['heading_error_rad']
        rate = result['projected_heading_rate_rad_s']
        quiet = abs(error) <= .02 and abs(rate) <= .1
        if self.terminal or abs(error) <= .02:
            self.integral_command = 0.
        else:
            if self.previous_error is not None and error*self.previous_error < 0:
                self.integral_command = 0.
            # Do not integrate a saturated command farther into saturation.
            raw = 1.5*error+self.integral_command
            if abs(raw) < .35 or raw*error < 0:
                self.integral_command = float(np.clip(
                    self.integral_command+.4*error*.1, -.12, .12))
        yaw = 0. if self.terminal or quiet else float(np.clip(
            1.5*error+self.integral_command, -.35, .35))
        result.update(requested_command=[0., 0., yaw],
                      integral_command_rad_s=self.integral_command,
                      controller='bounded_integral_alignment_development_v1')
        self.previous_error = error
        return result


def approach_line_command(position, rotation, origin, direction, up):
    """Observed relative line guidance; no division by remaining goal distance."""
    position, rotation, origin, direction, up = [np.asarray(v, dtype=float)
        for v in (position, rotation, origin, direction, up)]
    if (any(v.shape != (3,) or not np.isfinite(v).all()
            for v in (position, origin, direction, up))
            or rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-7, rtol=0)
            or abs(np.linalg.det(rotation)-1.) > 1e-7
            or abs(np.linalg.norm(up)-1.) > 1e-7):
        raise SensorContractError('finite observed line and proper attitude required')
    direction = direction-up*np.dot(direction, up)
    forward = rotation[:, 0]-up*np.dot(rotation[:, 0], up)
    if min(np.linalg.norm(direction), np.linalg.norm(forward)) < .8:
        raise SensorContractError('usable gravity-plane approach required')
    direction /= np.linalg.norm(direction)
    forward /= np.linalg.norm(forward)
    left = np.cross(up, direction)
    cross_track = float(np.dot(position-origin, left))
    # 0.5 m spatial response length, independent of endpoint proximity.
    desired = direction-left*np.clip(cross_track/.5, -.5, .5)
    desired /= np.linalg.norm(desired)
    heading_error = math.atan2(float(np.dot(up, np.cross(forward, desired))),
                              float(np.dot(forward, desired)))
    return {'yaw_command_rad_s': float(np.clip(1.5*heading_error, -.25, .25)),
            'cross_track_m': cross_track, 'heading_error_rad': heading_error,
            'lookahead_m': .5, 'clearance_qualified': False}


class MeasuredLineTraversal(MeasuredRegionTraversal):
    def __init__(self, geometry, observations):
        super().__init__(geometry, observations)
        self.approach_origin = self.approach_direction = None

    def _observe(self, packet, fast_packet, *, now_ns):
        result = super()._observe(packet, fast_packet, now_ns=now_ns)
        memory = self.observations.memory
        if self.target is not None and self.approach_origin is None:
            self.approach_origin = memory.position.copy()
            self.approach_direction = memory.rotation[:, 0].copy()
        if result['status'] == 'TRAVERSING':
            guidance = approach_line_command(memory.position, memory.rotation,
                self.approach_origin, self.approach_direction, memory.up_initial)
            result['requested_command'][2] = guidance['yaw_command_rad_s']
            result['line_guidance'] = guidance
        result['local_controller'] = 'measured_line_traversal_development_v1'
        return result


class MeasuredLineIntegralNavigation(MeasuredRegionNavigation):
    def __init__(self, method, geometry, template=None, *, memory_arm):
        if method != 'measured_line_integral' or template is not None:
            raise ValueError('distinct line/integral development baseline required')
        super().__init__('measured_region', geometry, memory_arm=memory_arm)

    def observe_rgbd(self, packet, fast_packet, depth, relative, *, now_ns):
        decision = super().observe_rgbd(packet, fast_packet, depth, relative, now_ns=now_ns)
        # Replace only newly constructed, unobserved operators. Never discard
        # an active controller's clocks, accumulated evidence or terminal state.
        if self.stage == 'TRAVERSE' and not isinstance(self.child, MeasuredLineTraversal):
            if self.child is None or self.child.tick != -1:
                raise SensorContractError('only an unstarted local operator may be replaced')
            self.child = MeasuredLineTraversal(self.geometry, self.regions)
            self.children[-1] = self.child
        if self.stage == 'ALIGN' and not isinstance(self.alignment, BoundedIntegralAlignment):
            if self.alignment is None or self.alignment.start_ns is not None:
                raise SensorContractError('only an unstarted alignment may be replaced')
            self.alignment = BoundedIntegralAlignment(self.selected['direction_initial_body'])
        decision['local_controller'] = 'measured_line_integral_navigation_development_v1'
        return decision
