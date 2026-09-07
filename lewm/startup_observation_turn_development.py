"""Setup-conditioned observation turn, not maze navigation or hardware safety.

All-posture URDF radius avoids pretending measured joint history bounds a future
gait. Future base translation is conditional on the explicit development speed
cap and stop horizon; a physical trial must independently check those assumptions.
"""
from copy import deepcopy
import math

import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_sensor_state import SensorContractError
from lewm.setup_region_prior_development import SetupRegionPrior
from lewm.setup_velocity_prior_development import SetupVelocityPrior, SetupVelocityPlaneMemory


MAX_BASE_SPEED_M_S = .3
COMMAND_PLUS_STOP_HORIZON_S = .4
PADDING_M = .04
MAX_YAW_COMMAND_RAD_S = .35
MAX_HEADING_RAD = .45


def all_posture_body_radius(geometry):
    """Triangle-inequality sphere for every joint angle and body orientation.

Each rigid-tree translation preserves its norm under all intervening rotations.
A primitive's centre offset plus circumsphere bounds every point on that shape.
This ignores joint limits and may be conservative; no posture sampling defines
the bound. The sphere is about the measured root/body origin, not the COM.
"""
    if not isinstance(geometry, ArticulatedCollisionGeometry): raise SensorContractError('verified articulated geometry required')
    radii = {'base': 0.}; pending = list(geometry._joints)
    while pending:
        ready = [j for j in pending if j['parent'] in radii]
        if not ready: raise SensorContractError('connected robot tree required')
        for j in ready:
            radii[j['child']] = radii[j['parent']] + float(np.linalg.norm(j['origin'][:3, 3]))
            pending.remove(j)
    bounds = {}
    for s in geometry._shapes:
        size = s['dimensions']
        if s['kind'] == 'box': extent = np.linalg.norm(size / 2)
        elif s['kind'] == 'cylinder': extent = np.hypot(size[0], size[1] / 2)
        elif s['kind'] == 'sphere': extent = size[0]
        else: raise SensorContractError('reviewed physical primitives required')
        bounds[s['shape_id']] = radii[s['link']] + float(np.linalg.norm(s['origin'][:3, 3])) + float(extent)
    maximum = max(bounds.values()) + 1e-12
    if not np.isfinite(maximum): raise SensorContractError('finite all-posture extent required')
    return dict(radius_m=maximum, per_shape_radius_m=bounds, all_joint_angles=True,
                root_frame='body_origin', dynamic_base_motion_included=False)


def validate_admission(admission, velocity_prior, region_prior):
    """A narrow setup-runner handoff, not a sensor observation or signature.

The trusted evaluator must bind checks_sha256 to its saved raw-check report.
This shape/type check cannot authenticate a forged report or validate hardware.
"""
    keys = {'schema', 'identity', 'anchor_ns', 'definition_sha256', 'checks_sha256',
            'velocity_and_nonfloor_checks_pass', 'initial_native_support_witness_present'}
    if (not isinstance(velocity_prior, SetupVelocityPrior) or not isinstance(region_prior, SetupRegionPrior)
            or not isinstance(admission, dict) or set(admission) != keys
            or admission['schema'] != 'startup_setup_admission_development.v1'
            or type(admission['anchor_ns']) is not int
            or admission['anchor_ns'] != velocity_prior.anchor_ns or admission['anchor_ns'] != region_prior.anchor_ns
            or not isinstance(admission['identity'], tuple) or any(type(x) is not int for x in admission['identity'])
            or admission['identity'] != velocity_prior.identity or admission['identity'] != region_prior.identity
            or admission['definition_sha256'] != velocity_prior.setup_evidence_sha256
            or admission['definition_sha256'] != region_prior.setup_evidence_sha256
            or admission['velocity_and_nonfloor_checks_pass'] is not True
            or admission['initial_native_support_witness_present'] is not True
            or not isinstance(admission['checks_sha256'], str) or len(admission['checks_sha256']) != 64
            or any(c not in '0123456789abcdef' for c in admission['checks_sha256'])):
        raise SensorContractError('exact checked setup handoff required before any startup action')
    return deepcopy(admission)


class StartupObservationTurn:
    """Observe full-rank depth, brake, and require three quiet rank-3 frames.

Uses the unchanged fusion budget and observed contradiction vetoes. A permitted
development command is conditional on checked static setup and future base-speed
assumptions; navigation/contact/dynamics/real-time qualification stays false.
"""
    def __init__(self, geometry, *, velocity_prior, region_prior, admission):
        self.admission = validate_admission(admission, velocity_prior, region_prior)
        self.region = region_prior
        self.radius = all_posture_body_radius(geometry)['radius_m']
        self.memory = SetupVelocityPlaneMemory(geometry, prior=velocity_prior, normal_error=.002,
            up_error=.001, plane_offset_error=.001, range_error_m=.001, beam_backend='compiled')
        self.status = 'NEW'; self.last_ns = None; self.brake_start_ns = None; self.quiet_rank3 = 0
        self.first_rank3_ns = None; self.last = None

    def observe(self, policy, depth, relative, *, now_ns):
        if self.status.startswith('FAILED_') or self.status == 'COMPLETE_OBSERVATION_TURN':
            raise SensorContractError('terminal startup controller; no restart')
        row = dict(decision_ns=now_ns, requested_command=[0., 0., 0.], terminal=False,
            evidence_role='SETUP_CONDITIONED_SENSOR_CONTROLLER', setup_checks_sha256=self.admission['checks_sha256'],
            first_rank3_ns=self.first_rank3_ns, contact_permitted=False, navigation_qualified=False,
            future_dynamics_validated=False, real_time_qualified=False)
        try:
            self.memory.observe(policy, depth, relative, now_ns=now_ns)
            evidence = self.memory.query_current_primitives(now_ns=now_ns)
            fusion = self.memory._rays.fusion
            centre = np.asarray(fusion['position_initial_body_m'])
            extent = self.radius + PADDING_M + MAX_BASE_SPEED_M_S * COMMAND_PLUS_STOP_HORIZON_S + fusion['position_error_scale_m']
            conflicts = bool(np.any(evidence['non_floor_conflict'] | evidence['floor_penetration']))
            through_ns = now_ns + int(round(COMMAND_PLUS_STOP_HORIZON_S * 1e9))
            region = self.region.query([centre-extent], [centre+extent], [0.],
                identity=self.memory._rays.identity, now_ns=through_ns, observed_conflict=[conflicts])
            rotation = np.asarray(relative['relative_orientation']['rotation_initial_body_from_current_body'])
            forward = rotation[:, 0]
            if np.linalg.norm(forward[:2]) < .2: raise SensorContractError('upright observed heading required')
            heading = math.atan2(forward[1], forward[0])
            gyro_norm = float(np.linalg.norm(policy['sensor_state']['sensed']['gyro']['values'][-1]))
            speed = float(np.linalg.norm(fusion['velocity_initial_body_m_s']))
            speed_with_prior = speed + fusion['initial_velocity_prior_transport']['velocity_radius_m_s']
            row.update(depth_rank=fusion['depth_rank'], relative_heading_rad=heading,
                measured_gyro_norm_rad_s=gyro_norm, inferred_speed_m_s=speed,
                inferred_speed_plus_prior_radius_m_s=speed_with_prior,
                combined_pose_scale_m=fusion['position_error_scale_m'],
                body_all_posture_radius_m=self.radius, command_plus_stop_extent_m=extent,
                envelope_through_ns=through_ns,
                region_conditionally_contains_envelope=bool(region['conditional_setup_non_floor_clearance'][0]),
                observed_conflict=conflicts)
            if conflicts: self.status = 'FAILED_OBSERVED_CONFLICT'
            elif speed_with_prior > MAX_BASE_SPEED_M_S: self.status = 'FAILED_INFERRED_BASE_SPEED'
            elif not region['conditional_setup_non_floor_clearance'][0]: self.status = 'FAILED_SETUP_ENVELOPE_OR_EXPIRY'
            elif abs(heading) > MAX_HEADING_RAD: self.status = 'FAILED_HEADING_LIMIT'
            else:
                rank3 = fusion['depth_rank'] == 3
                if rank3 and self.first_rank3_ns is None:
                    self.first_rank3_ns = now_ns; self.brake_start_ns = now_ns
                if self.brake_start_ns is None:
                    self.status = 'OBSERVATION_TURN'
                    row['requested_command'] = [0., 0., MAX_YAW_COMMAND_RAD_S]
                else:
                    self.status = 'BRAKING_FOR_OBSERVATION'
                    self.quiet_rank3 = self.quiet_rank3 + 1 if rank3 and gyro_norm <= .1 and speed <= .05 else 0
                    if now_ns - self.brake_start_ns >= 300_000_000 and self.quiet_rank3 >= 3:
                        self.status = 'COMPLETE_OBSERVATION_TURN'
                row['first_rank3_ns'] = self.first_rank3_ns
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            self.status = 'FAILED_SENSOR_OR_FUSION'
            reasons = []
            while error is not None:
                reasons.append(str(error)); error = error.__cause__
            row['failure_chain'] = reasons
        terminal = self.status.startswith('FAILED_') or self.status == 'COMPLETE_OBSERVATION_TURN'
        if terminal: row['requested_command'] = [0., 0., 0.]
        row.update(status=self.status, terminal=terminal, quiet_rank3_frames=self.quiet_rank3)
        self.last_ns = now_ns; self.last = deepcopy(row)
        return row
