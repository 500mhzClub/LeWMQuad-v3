"""Finite, expiring setup-only non-floor-clear region; never observed free space.

The prism is expressed in the INITIAL BODY frame. The caller must transport
query boxes into that frame and include all pose, geometry and motion-envelope
errors. This contract does not define or validate supporting ground/contact.
"""
from dataclasses import dataclass

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.setup_velocity_prior_development import SetupVelocityPrior


@dataclass(frozen=True)
class SetupRegionPrior:
    identity: tuple
    anchor_ns: int
    valid_until_ns: int
    lower_initial_body_m: tuple
    upper_initial_body_m: tuple
    setup_evidence_sha256: str

    def __post_init__(self):
        # Reuse the strict episode/epoch/evidence-reference validation only.
        SetupVelocityPrior(self.identity, self.anchor_ns, (0., 0., 0.), 1., self.setup_evidence_sha256)
        low, high = self.lower_initial_body_m, self.upper_initial_body_m
        if (type(self.valid_until_ns) is not int or self.valid_until_ns <= self.anchor_ns
                or not isinstance(low, tuple) or not isinstance(high, tuple)
                or len(low) != 3 or len(high) != 3
                or any(type(x) not in (int, float) for x in (*low, *high))
                or not np.isfinite([low, high]).all() or np.any(np.asarray(low) >= high)):
            raise SensorContractError('finite ordered immutable initial-body prism and finite validity interval required')

    def query(self, lower_initial_body_m, upper_initial_body_m, point_radius_m,
              *, identity, now_ns, observed_conflict):
        low, high = np.asarray(lower_initial_body_m, float), np.asarray(upper_initial_body_m, float)
        radius, conflict = np.asarray(point_radius_m, float), np.asarray(observed_conflict)
        if (low.ndim != 2 or low.shape[1:] != (3,) or high.shape != low.shape
                or radius.shape != (len(low),) or conflict.shape != radius.shape or conflict.dtype != bool
                or not np.isfinite([low, high]).all() or not np.isfinite(radius).all()
                or np.any(low > high) or np.any(radius < 0) or type(now_ns) is not int
                or not isinstance(identity, tuple) or any(type(x) is not int for x in identity)
                or identity != self.identity):
            raise SensorContractError('aligned finite initial-frame query/error/conflict and exact episode required')
        active = self.anchor_ns <= now_ns <= self.valid_until_ns
        with np.errstate(over='ignore', invalid='ignore'):
            lo, hi = low - radius[:, None], high + radius[:, None]
        if not np.isfinite([lo, hi]).all(): raise SensorContractError('representable uncertainty-expanded prism required')
        # Strict interior guards against roundoff at the supplied region edge.
        margin = 1e-12 + 128 * np.finfo(float).eps * np.maximum(np.abs(lo), np.abs(hi))
        inside = (lo - margin >= self.lower_initial_body_m).all(axis=1) & (hi + margin <= self.upper_initial_body_m).all(axis=1)
        return dict(conditional_setup_non_floor_clearance=active & inside & ~conflict,
            observed_conflict=conflict.copy(), region_active=active,
            evidence_role='SUPPLIED_SETUP_REGION_NOT_SENSOR', setup_evidence_sha256=self.setup_evidence_sha256,
            setup_condition_assumed=True, setup_independently_verified=False,
            observed_free_space=False, support_established=False, contact_permitted=False,
            future_gait_qualified=False, navigation_qualified=False)
