"""Explicit setup velocity assumption with conditional weak-subspace transport.

No zero-velocity detection, support inference, setup verification or sensor-error
calibration is performed here. Depth ranks remain unchanged. The prior's ball
and the inherited uncalibrated error proxy are reported separately.
"""
from dataclasses import dataclass, asdict

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.depth_inertial_moment_fusion_development import MomentWeakSubspaceIntegrator, ASSUMPTIONS
from lewm.measured_plane_obstacle_memory_development import MeasuredPlaneObstacleMemory
from lewm.uncertain_ray_memory_development import FusedRayEvidenceMemory


@dataclass(frozen=True)
class SetupVelocityPrior:
    identity: tuple
    anchor_ns: int
    mean_initial_body_m_s: tuple
    radius_m_s: float
    setup_evidence_sha256: str

    def __post_init__(self):
        if (not isinstance(self.identity, tuple) or len(self.identity) != 3
                or any(type(x) is not int or x < 0 for x in self.identity)
                or type(self.anchor_ns) is not int or self.anchor_ns < 0
                or not isinstance(self.mean_initial_body_m_s, tuple)
                or len(self.mean_initial_body_m_s) != 3
                or any(type(x) not in (int, float) for x in self.mean_initial_body_m_s)
                or not np.isfinite(self.mean_initial_body_m_s).all()
                or type(self.radius_m_s) not in (int, float)
                or not np.isfinite(self.radius_m_s) or self.radius_m_s <= 0
                or not isinstance(self.setup_evidence_sha256, str)
                or len(self.setup_evidence_sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.setup_evidence_sha256)):
            raise SensorContractError('immutable finite nonzero-radius setup prior with exact identity/epoch/evidence reference required')


class PriorVelocitySensitivity:
    """Exact linear dependence on initial velocity, conditioned on depth/gyro.

For a weak-space projector P in initial-body coordinates, V'=PV and
X'=X+dt PV. A fully observed displacement sets V'=0, preserving X.
These factors do not cover errors in P, acceleration, depth or calibration.
"""
    def __init__(self):
        self.position_factor = np.zeros((3, 3))
        self.velocity_factor = np.eye(3)

    def advance(self, projector, dt):
        p = np.asarray(projector, float)
        if (p.shape != (3, 3) or not np.isfinite(p).all()
                or not np.allclose(p, p.T, atol=1e-7, rtol=0)
                or not np.allclose(p @ p, p, atol=1e-7, rtol=0)
                or type(dt) not in (int, float) or not np.isfinite(dt) or dt <= 0):
            raise SensorContractError('finite orthogonal weak projector and positive interval required')
        v = p @ self.velocity_factor
        x = self.position_factor + dt * v
        if not np.isfinite([x, v]).all():
            raise SensorContractError('representable prior transport required')
        self.position_factor, self.velocity_factor = x, v

    def snapshot(self, radius):
        if not np.isfinite(radius) or radius <= 0:
            raise SensorContractError('positive finite initial velocity radius required')
        bounds = []
        for matrix in (self.position_factor, self.velocity_factor):
            norm = float(np.linalg.norm(matrix, ord=2))
            value = radius * norm
            if not np.isfinite(value): raise SensorContractError('representable prior radius required')
            # Explicit numerical allowance, not an extra physical noise source.
            bounds.append(0. if norm == 0 else value + 1e-12 * (1 + value))
        return dict(position_factor_s=self.position_factor.tolist(),
            velocity_factor=self.velocity_factor.tolist(),
            position_radius_m=bounds[0], velocity_radius_m_s=bounds[1],
            conditioned_on_depth_subspaces_and_gyro=True, sensor_error_bound=False)


class SetupVelocityIntegrator(MomentWeakSubspaceIntegrator):
    """Initialize ONLY from a supplied setup prior, never an applied command."""
    def __init__(self, prior):
        if not isinstance(prior, SetupVelocityPrior):
            raise SensorContractError('explicit setup prior required; no implicit zero velocity')
        super().__init__()
        self.prior = prior
        self.prior_sensitivity = PriorVelocitySensitivity()

    def observe(self, policy, depth_state):
        if self.failed: raise SensorContractError('setup-prior fusion fault latched')
        try:
            first = self.last_ns is None
            previous_rotation = None if first else self.rotation.copy()
            if first:
                if (policy['sensor_state']['decision_ns'] != self.prior.anchor_ns
                        or tuple(policy['sensor_state']['identity']) != self.prior.identity):
                    raise SensorContractError('setup prior epoch or episode mismatch')
                self.velocity = np.asarray(self.prior.mean_initial_body_m_s, float).copy()
            row = super().observe(policy, depth_state)
            if not first:
                weak = np.asarray(depth_state['motion']['weak_directions_previous_body'], float).reshape(-1, 3)
                w = weak @ previous_rotation.T
                self.prior_sensitivity.advance(w.T @ w, .1)
            sensitivity = self.prior_sensitivity.snapshot(self.prior.radius_m_s)
            inherited = row['position_error_scale_m']
            combined = inherited + sensitivity['position_radius_m']
            if not np.isfinite(combined): raise SensorContractError('finite combined development scale required')
            return row | dict(schema='setup_velocity_moment_fusion_development.v1',
                initial_velocity_prior=asdict(self.prior), initial_velocity_source='SUPPLIED_SETUP_PRIOR_NOT_SENSOR',
                initial_velocity_prior_transport=sensitivity,
                inherited_position_error_scale_m=inherited,
                position_error_scale_m=combined,
                usable_under_declared_proxy_budget=bool(combined <= ASSUMPTIONS['maximum_position_scale_m']),
                scale_composition='inherited_uncalibrated_proxy_plus_conditional_setup_ball_radius',
                setup_condition_assumed=True, setup_independently_verified=False,
                contact_permitted=False, navigation_qualified=False)
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            self.failed = True
            raise SensorContractError('invalid setup-prior fusion; stop') from error


class SetupVelocityRayMemory(FusedRayEvidenceMemory):
    def __init__(self, prior):
        super().__init__()
        self.integrator = SetupVelocityIntegrator(prior)


class SetupVelocityPlaneMemory(MeasuredPlaneObstacleMemory):
    """Inherited view transport consumes the combined scale; no floor is added."""
    def __init__(self, geometry, *, prior, **kwargs):
        super().__init__(geometry, **kwargs)
        self._rays = SetupVelocityRayMemory(prior)
